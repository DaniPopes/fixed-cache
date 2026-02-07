#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::new_without_default)]

use equivalent::Equivalent;
use std::{
    cell::UnsafeCell,
    convert::Infallible,
    fmt,
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "stats")]
mod stats;
#[cfg(feature = "stats")]
#[cfg_attr(docsrs, doc(cfg(feature = "stats")))]
pub use stats::{AnyRef, CountingStatsHandler, Stats, StatsHandler};

// Tag bit layout (64-bit):
//
//   63      56 55                              12 11        2  1  0
//  +----------+----------------------------------+----------+--+--+
//  |  version |          hash signature          |   epoch  | A| L|
//  |  (8-bit) |  (variable, capacity-dependent)  | (10-bit) |  |  |
//  +----------+----------------------------------+----------+--+--+
//
// L (locked): Set during writes. Readers (seqlock path) skip the bucket; locked-path readers
//   fail the CAS and fall through to a miss. Writers fail the CAS and abandon the insert.
// A (alive): Indicates the bucket contains initialized data. Cleared on remove, set on insert.
//   Part of the tag identity: readers require it to match, so empty buckets are never "hit".
// epoch: Tracks which epoch the entry belongs to, enabling O(1) cache invalidation via
//   `Cache::clear`. Only present when `CacheConfig::EPOCHS` is true; otherwise these bits
//   are part of the hash signature.
// hash signature: Upper bits of the key's hash, used to reject non-matching keys without
//   reading the bucket data. The exact width depends on the cache capacity (more buckets =
//   fewer signature bits).
// version: Monotonic counter incremented on every mutation (insert/remove). Used by the
//   seqlock read path to detect concurrent writes: the reader snapshots the tag, speculatively
//   reads the data, then verifies the tag hasn't changed. The 8-bit counter makes ABA
//   wraparound (256 writes between two reader loads) effectively impossible in practice.

const LOCKED_BIT: usize = 1 << 0;
const ALIVE_BIT: usize = 1 << 1;
const NEEDED_BITS: usize = 2;

const EPOCH_BITS: usize = 10;
const EPOCH_SHIFT: usize = NEEDED_BITS;
const EPOCH_MASK: usize = ((1 << EPOCH_BITS) - 1) << EPOCH_SHIFT;
const EPOCH_NEEDED_BITS: usize = NEEDED_BITS + EPOCH_BITS;

const VERSION_BITS: u32 = 8;
const VERSION_SHIFT: u32 = usize::BITS - VERSION_BITS;
const VERSION_MASK: usize = ((1usize << VERSION_BITS) - 1) << VERSION_SHIFT;
const VERSION_INCREMENT: usize = 1usize << VERSION_SHIFT;

#[cfg(feature = "rapidhash")]
type DefaultBuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
#[cfg(not(feature = "rapidhash"))]
type DefaultBuildHasher = std::hash::RandomState;

/// Configuration trait for [`Cache`].
///
/// This trait allows customizing cache behavior through compile-time configuration.
pub trait CacheConfig {
    /// Whether to track statistics for cache performance.
    ///
    /// When enabled, the cache tracks hit and miss counts for each key, allowing
    /// monitoring of cache performance.
    ///
    /// Only enabled when the `stats` feature is also enabled.
    ///
    /// Defaults to `true`.
    const STATS: bool = true;

    /// Whether to track epochs for cheap invalidation via [`Cache::clear`].
    ///
    /// When enabled, the cache tracks an epoch counter that is incremented on each call to
    /// [`Cache::clear`]. Entries are considered invalid if their epoch doesn't match the current
    /// epoch, allowing O(1) invalidation of all entries without touching each bucket.
    ///
    /// Defaults to `false`.
    const EPOCHS: bool = false;
}

/// Default cache configuration.
pub struct DefaultCacheConfig(());
impl CacheConfig for DefaultCacheConfig {}

/// A concurrent, fixed-size, set-associative cache.
///
/// This cache maps keys to values using a fixed number of buckets. Each key hashes to exactly
/// one bucket, and collisions are resolved by eviction (the new value replaces the old one).
///
/// # Thread Safety
///
/// The cache is safe to share across threads (`Send + Sync`). All operations use atomic
/// instructions and never block, making it suitable for high-contention scenarios.
///
/// # Limitations
///
/// - **Eviction on collision**: When two keys hash to the same bucket, the older entry is evicted.
/// - **No iteration**: Individual entries cannot be enumerated.
///
/// # Type Parameters
///
/// - `K`: The key type. Must implement [`Hash`] + [`Eq`].
/// - `V`: The value type. Must implement [`Clone`].
/// - `S`: The hash builder type. Must implement [`BuildHasher`]. Defaults to [`RandomState`] or
///   [`rapidhash`] if the `rapidhash` feature is enabled.
///
/// # Example
///
/// ```
/// use fixed_cache::Cache;
///
/// let cache: Cache<u64, u64> = Cache::new(256, Default::default());
///
/// // Insert a value
/// cache.insert(42, 100);
/// assert_eq!(cache.get(&42), Some(100));
///
/// // Get or compute a value
/// let value = cache.get_or_insert_with(123, |&k| k * 2);
/// assert_eq!(value, 246);
/// ```
///
/// [`Hash`]: std::hash::Hash
/// [`Eq`]: std::cmp::Eq
/// [`Clone`]: std::clone::Clone
/// [`Drop`]: std::ops::Drop
/// [`BuildHasher`]: std::hash::BuildHasher
/// [`RandomState`]: std::hash::RandomState
/// [`rapidhash`]: https://crates.io/crates/rapidhash
pub struct Cache<K, V, S = DefaultBuildHasher, C: CacheConfig = DefaultCacheConfig> {
    entries: *const [Bucket<(K, V)>],
    build_hasher: S,
    #[cfg(feature = "stats")]
    stats: Option<Stats<K, V>>,
    drop: bool,
    epoch: AtomicUsize,
    _config: PhantomData<C>,
}

impl<K, V, S, C: CacheConfig> fmt::Debug for Cache<K, V, S, C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cache").finish_non_exhaustive()
    }
}

// SAFETY: `Cache` is safe to share across threads because `Bucket` uses atomic operations.
unsafe impl<K: Send, V: Send, S: Send, C: CacheConfig + Send> Send for Cache<K, V, S, C> {}
unsafe impl<K: Send, V: Send, S: Sync, C: CacheConfig + Sync> Sync for Cache<K, V, S, C> {}

impl<K, V, S, C> Cache<K, V, S, C>
where
    K: Hash + Eq,
    S: BuildHasher,
    C: CacheConfig,
{
    const NEEDS_DROP: bool = Bucket::<(K, V)>::NEEDS_DROP;

    // TODO: Not entirely correct.
    const ENTRY_IMPLS_COPY: bool = !Self::NEEDS_DROP;

    /// Create a new cache with the specified number of entries and hasher.
    ///
    /// Dynamically allocates memory for the cache entries.
    ///
    /// # Panics
    ///
    /// Panics if `num`:
    /// - is not a power of two.
    /// - isn't at least 4.
    // See len_assertion for why.
    pub fn new(num: usize, build_hasher: S) -> Self {
        Self::len_assertion(num);
        let layout = std::alloc::Layout::array::<Bucket<(K, V)>>(num).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let entries = std::ptr::slice_from_raw_parts(ptr.cast::<Bucket<(K, V)>>(), num);
        Self::new_inner(entries, build_hasher, true)
    }

    /// Creates a new cache with the specified entries and hasher.
    ///
    /// # Panics
    ///
    /// See [`new`](Self::new).
    #[inline]
    pub const fn new_static(entries: &'static [Bucket<(K, V)>], build_hasher: S) -> Self {
        Self::len_assertion(entries.len());
        Self::new_inner(entries, build_hasher, false)
    }

    /// Sets the cache's statistics.
    ///
    /// Can only be called when [`CacheConfig::STATS`] is `true`.
    #[cfg(feature = "stats")]
    #[inline]
    pub fn with_stats(mut self, stats: Option<Stats<K, V>>) -> Self {
        self.set_stats(stats);
        self
    }

    /// Sets the cache's statistics.
    ///
    /// Can only be called when [`CacheConfig::STATS`] is `true`.
    #[cfg(feature = "stats")]
    #[inline]
    pub fn set_stats(&mut self, stats: Option<Stats<K, V>>) {
        const { assert!(C::STATS, "can only set stats when C::STATS is true") }
        self.stats = stats;
    }

    #[inline]
    const fn new_inner(entries: *const [Bucket<(K, V)>], build_hasher: S, drop: bool) -> Self {
        Self {
            entries,
            build_hasher,
            #[cfg(feature = "stats")]
            stats: None,
            drop,
            epoch: AtomicUsize::new(0),
            _config: PhantomData,
        }
    }

    #[inline]
    const fn len_assertion(len: usize) {
        // We need `RESERVED_BITS` bits to store metadata for each entry.
        // Since we calculate the tag mask based on the index mask, and the index mask is (len - 1),
        // we assert that the length's bottom `RESERVED_BITS` bits are zero.
        let reserved = if C::EPOCHS { EPOCH_NEEDED_BITS } else { NEEDED_BITS };
        assert!(len.is_power_of_two(), "length must be a power of two");
        assert!((len & ((1 << reserved) - 1)) == 0, "len must have its bottom N bits set to zero");
    }

    #[inline]
    const fn index_mask(&self) -> usize {
        let n = self.capacity();
        unsafe { std::hint::assert_unchecked(n.is_power_of_two()) };
        n - 1
    }

    #[inline]
    const fn tag_mask(&self) -> usize {
        !self.index_mask()
    }

    /// Returns the current epoch of the cache.
    ///
    /// Only meaningful when [`CacheConfig::EPOCHS`] is `true`.
    #[inline]
    fn epoch(&self) -> usize {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Clears the cache by invalidating all entries.
    ///
    /// This is O(1): it simply increments an epoch counter. On epoch wraparound, falls back to
    /// [`clear_slow`](Self::clear_slow).
    ///
    /// Can only be called when [`CacheConfig::EPOCHS`] is `true`.
    #[inline]
    pub fn clear(&self) {
        const EPOCH_MAX: usize = (1 << EPOCH_BITS) - 1;
        const { assert!(C::EPOCHS, "can only .clear() when C::EPOCHS is true") }
        let prev = self.epoch.fetch_add(1, Ordering::Release);
        if prev == EPOCH_MAX {
            self.clear_slow();
        }
    }

    /// Clears the cache by zeroing all bucket memory.
    ///
    /// This is O(N) where N is the number of buckets. Prefer [`clear`](Self::clear) when
    /// [`CacheConfig::EPOCHS`] is `true`.
    ///
    /// # Safety
    ///
    /// This method is safe but may race with concurrent operations. Callers should ensure
    /// no other threads are accessing the cache during this operation.
    pub fn clear_slow(&self) {
        // SAFETY: We zero the entire bucket array. This is safe because:
        // - Bucket<T> is repr(C) and zeroed memory is a valid empty bucket state.
        // - We don't need to drop existing entries since we're zeroing the ALIVE_BIT.
        // - Concurrent readers will see either the old state or zeros (empty).
        unsafe {
            std::ptr::write_bytes(
                self.entries.cast_mut().cast::<Bucket<(K, V)>>(),
                0,
                self.entries.len(),
            )
        };
    }

    /// Returns the hash builder used by this cache.
    #[inline]
    pub const fn hasher(&self) -> &S {
        &self.build_hasher
    }

    /// Returns the number of entries in this cache.
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.entries.len()
    }

    /// Returns the statistics of this cache.
    #[cfg(feature = "stats")]
    pub const fn stats(&self) -> Option<&Stats<K, V>> {
        self.stats.as_ref()
    }
}

impl<K, V, S, C> Cache<K, V, S, C>
where
    K: Hash + Eq,
    V: Clone,
    S: BuildHasher,
    C: CacheConfig,
{
    /// Get an entry from the cache.
    pub fn get<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> Option<V> {
        let (bucket, tag) = self.calc(key);
        self.get_inner(key, bucket, tag)
    }

    #[inline]
    fn get_inner<Q: ?Sized + Hash + Equivalent<K>>(
        &self,
        key: &Q,
        bucket: &Bucket<(K, V)>,
        tag: usize,
    ) -> Option<V> {
        if Self::ENTRY_IMPLS_COPY {
            // Seqlock fast path: speculatively read the tag, then the data, then re-read
            // the tag. If the tag hasn't changed between the two reads (including the version
            // counter), no writer intervened and the data is consistent. This avoids acquiring
            // the lock entirely on cache hits, at the cost of occasionally discarding a read
            // if a concurrent write raced with us.
            let seqlock = bucket.tag.load(Ordering::Acquire);
            if (seqlock & LOCKED_BIT) == 0 && (seqlock & !VERSION_MASK) == tag {
                let (ck, v) = unsafe { bucket.data.get().cast::<(K, V)>().read() };
                if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
                    std::sync::atomic::compiler_fence(Ordering::Acquire);
                } else {
                    std::sync::atomic::fence(Ordering::Acquire);
                }
                if seqlock == bucket.tag.load(Ordering::Acquire) && key.equivalent(&ck) {
                    #[cfg(feature = "stats")]
                    if C::STATS
                        && let Some(stats) = &self.stats
                    {
                        stats.record_hit(&ck, &v);
                    }
                    return Some(v);
                }
            }
        } else if let Some(prev) = bucket.try_lock_ret(Some(tag), false) {
            // SAFETY: We hold the lock and bucket is alive, so we have exclusive access.
            let (ck, v) = unsafe { (*bucket.data.get()).assume_init_ref() };
            if key.equivalent(ck) {
                #[cfg(feature = "stats")]
                if C::STATS
                    && let Some(stats) = &self.stats
                {
                    stats.record_hit(ck, v);
                }
                let v = v.clone();
                bucket.unlock(prev);
                return Some(v);
            }
            bucket.unlock(prev);
        }
        #[cfg(feature = "stats")]
        if C::STATS
            && let Some(stats) = &self.stats
        {
            stats.record_miss(AnyRef::new(&key));
        }
        None
    }

    /// Insert an entry into the cache.
    pub fn insert(&self, key: K, value: V) {
        let (bucket, tag) = self.calc(&key);
        self.insert_inner(bucket, tag, || (key, value));
    }

    /// Remove an entry from the cache.
    ///
    /// Returns the value if the key was present in the cache.
    pub fn remove<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> Option<V> {
        let (bucket, tag) = self.calc(key);
        if let Some(prev) = bucket.try_lock_ret(Some(tag), false) {
            // SAFETY: We hold the lock and bucket is alive, so we have exclusive access.
            let data = unsafe { &mut *bucket.data.get() };
            let (ck, v) = unsafe { data.assume_init_ref() };
            if key.equivalent(ck) {
                let v = v.clone();
                #[cfg(feature = "stats")]
                if C::STATS
                    && let Some(stats) = &self.stats
                {
                    stats.record_remove(ck, &v);
                }
                if Self::NEEDS_DROP {
                    // SAFETY: We hold the lock, so we have exclusive access.
                    unsafe { data.assume_init_drop() };
                }
                let new_version = prev.wrapping_add(VERSION_INCREMENT) & VERSION_MASK;
                bucket.unlock(new_version);
                return Some(v);
            }
            bucket.unlock(prev);
        }
        None
    }

    #[inline]
    fn insert_inner(
        &self,
        bucket: &Bucket<(K, V)>,
        tag: usize,
        make_entry: impl FnOnce() -> (K, V),
    ) {
        #[inline(always)]
        unsafe fn do_write<T>(ptr: *mut T, f: impl FnOnce() -> T) {
            // This function is translated as:
            // - allocate space for a T on the stack
            // - call f() with the return value being put onto this stack space
            // - memcpy from the stack to the heap
            //
            // Ideally we want LLVM to always realize that doing a stack
            // allocation is unnecessary and optimize the code so it writes
            // directly into the heap instead. It seems we get it to realize
            // this most consistently if we put this critical line into it's
            // own function instead of inlining it into the surrounding code.
            unsafe { core::ptr::write(ptr, f()) };
        }

        let Some(locked) = bucket.try_lock_ret(None, true) else {
            return;
        };
        let to_store = tag | (locked & VERSION_MASK);

        // SAFETY: We hold the lock, so we have exclusive access.
        unsafe {
            let is_alive = locked & ALIVE_BIT != 0;
            let data = bucket.data.get().cast::<(K, V)>();

            if C::STATS && cfg!(feature = "stats") {
                #[cfg(feature = "stats")]
                if is_alive {
                    let (old_key, old_value) = std::ptr::replace(data, make_entry());
                    if let Some(stats) = &self.stats {
                        stats.record_insert(&(*data).0, &(*data).1, Some((&old_key, &old_value)));
                    }
                } else {
                    do_write(data, make_entry);
                    if let Some(stats) = &self.stats {
                        stats.record_insert(&(*data).0, &(*data).1, None);
                    }
                }
            } else {
                if Self::NEEDS_DROP && is_alive {
                    std::ptr::drop_in_place(data);
                }
                do_write(data, make_entry);
            }
        }
        bucket.unlock(to_store);
    }

    /// Gets a value from the cache, or inserts one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns a clone of the cached value.
    /// Otherwise, calls `f` to compute the value, attempts to insert it, and returns it.
    #[inline]
    pub fn get_or_insert_with<F>(&self, key: K, f: F) -> V
    where
        F: FnOnce(&K) -> V,
    {
        let Ok(v) = self.get_or_try_insert_with(key, |key| Ok::<_, Infallible>(f(key)));
        v
    }

    /// Gets a value from the cache, or inserts one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns a clone of the cached value.
    /// Otherwise, calls `f` to compute the value, attempts to insert it, and returns it.
    ///
    /// This is the same as [`get_or_insert_with`], but takes a reference to the key, and a function
    /// to get the key reference to an owned key.
    ///
    /// [`get_or_insert_with`]: Self::get_or_insert_with
    #[inline]
    pub fn get_or_insert_with_ref<'a, Q, F, Cvt>(&self, key: &'a Q, f: F, cvt: Cvt) -> V
    where
        Q: ?Sized + Hash + Equivalent<K>,
        F: FnOnce(&'a Q) -> V,
        Cvt: FnOnce(&'a Q) -> K,
    {
        let Ok(v) = self.get_or_try_insert_with_ref(key, |key| Ok::<_, Infallible>(f(key)), cvt);
        v
    }

    /// Gets a value from the cache, or attempts to insert one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns `Ok` with a clone of the cached value.
    /// Otherwise, calls `f` to compute the value. If `f` returns `Ok`, attempts to insert
    /// the value and returns it. If `f` returns `Err`, the error is propagated.
    #[inline]
    pub fn get_or_try_insert_with<F, E>(&self, key: K, f: F) -> Result<V, E>
    where
        F: FnOnce(&K) -> Result<V, E>,
    {
        let mut key = std::mem::ManuallyDrop::new(key);
        let mut read = false;
        let r = self.get_or_try_insert_with_ref(&*key, f, |k| {
            read = true;
            unsafe { std::ptr::read(k) }
        });
        if !read {
            unsafe { std::mem::ManuallyDrop::drop(&mut key) }
        }
        r
    }

    /// Gets a value from the cache, or attempts to insert one computed by `f` if not present.
    ///
    /// If the key is found in the cache, returns `Ok` with a clone of the cached value.
    /// Otherwise, calls `f` to compute the value. If `f` returns `Ok`, attempts to insert
    /// the value and returns it. If `f` returns `Err`, the error is propagated.
    ///
    /// This is the same as [`Self::get_or_try_insert_with`], but takes a reference to the key, and
    /// a function to get the key reference to an owned key.
    #[inline]
    pub fn get_or_try_insert_with_ref<'a, Q, F, Cvt, E>(
        &self,
        key: &'a Q,
        f: F,
        cvt: Cvt,
    ) -> Result<V, E>
    where
        Q: ?Sized + Hash + Equivalent<K>,
        F: FnOnce(&'a Q) -> Result<V, E>,
        Cvt: FnOnce(&'a Q) -> K,
    {
        let (bucket, tag) = self.calc(key);
        if let Some(v) = self.get_inner(key, bucket, tag) {
            return Ok(v);
        }
        let value = f(key)?;
        self.insert_inner(bucket, tag, || (cvt(key), value.clone()));
        Ok(value)
    }

    #[inline]
    fn calc<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> (&Bucket<(K, V)>, usize) {
        let hash = self.hash_key(key);
        // SAFETY: index is masked to be within bounds.
        let bucket = unsafe { (&*self.entries).get_unchecked(hash & self.index_mask()) };
        let mut tag = hash & self.tag_mask() & !VERSION_MASK | ALIVE_BIT;
        if C::EPOCHS {
            tag = (tag & !EPOCH_MASK) | ((self.epoch() << EPOCH_SHIFT) & EPOCH_MASK);
        }
        (bucket, tag)
    }

    #[inline]
    fn hash_key<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> usize {
        let hash = self.build_hasher.hash_one(key);

        if cfg!(target_pointer_width = "32") {
            ((hash >> 32) as usize) ^ (hash as usize)
        } else {
            hash as usize
        }
    }
}

impl<K, V, S, C: CacheConfig> Drop for Cache<K, V, S, C> {
    fn drop(&mut self) {
        if self.drop {
            // SAFETY: `Drop` has exclusive access.
            drop(unsafe { Box::from_raw(self.entries.cast_mut()) });
        }
    }
}

/// A single cache bucket that holds one key-value pair.
///
/// Buckets are aligned to 128 bytes to avoid false sharing between cache lines.
/// Each bucket contains an atomic tag for lock-free synchronization and uninitialized
/// storage for the data.
///
/// This type is public to allow use with the [`static_cache!`] macro for compile-time
/// cache initialization. You typically don't need to interact with it directly.
#[repr(C, align(128))]
#[doc(hidden)]
pub struct Bucket<T> {
    tag: AtomicUsize,
    data: UnsafeCell<MaybeUninit<T>>,
}

impl<T> Bucket<T> {
    const NEEDS_DROP: bool = std::mem::needs_drop::<T>();

    /// Creates a new zeroed bucket.
    #[inline]
    pub const fn new() -> Self {
        Self { tag: AtomicUsize::new(0), data: UnsafeCell::new(MaybeUninit::zeroed()) }
    }

    #[inline]
    fn try_lock_ret(&self, expected: Option<usize>, bump_version: bool) -> Option<usize> {
        let state = self.tag.load(Ordering::Relaxed);
        if let Some(expected) = expected {
            if state & !VERSION_MASK != expected {
                return None;
            }
        } else if state & LOCKED_BIT != 0 {
            return None;
        }
        let mut locked = state | LOCKED_BIT;
        if bump_version {
            let new_version = state.wrapping_add(VERSION_INCREMENT) & VERSION_MASK;
            locked = (locked & !VERSION_MASK) | new_version;
        }
        self.tag
            .compare_exchange(state, locked, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|prev| if bump_version { locked } else { prev })
    }

    #[inline]
    fn is_alive(&self) -> bool {
        self.tag.load(Ordering::Relaxed) & ALIVE_BIT != 0
    }

    #[inline]
    fn unlock(&self, tag: usize) {
        self.tag.store(tag, Ordering::Release);
    }
}

// SAFETY: `Bucket` is a specialized `Mutex<T>` that never blocks.
unsafe impl<T: Send> Send for Bucket<T> {}
unsafe impl<T: Send> Sync for Bucket<T> {}

impl<T> Drop for Bucket<T> {
    fn drop(&mut self) {
        if Self::NEEDS_DROP && self.is_alive() {
            // SAFETY: `Drop` has exclusive access.
            unsafe { self.data.get_mut().assume_init_drop() };
        }
    }
}

/// Declares a static cache with the given name, key type, value type, and size.
///
/// The size must be a power of two.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "rapidhash")] {
/// use fixed_cache::{Cache, static_cache};
///
/// type BuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
///
/// static MY_CACHE: Cache<u64, &'static str, BuildHasher> =
///     static_cache!(u64, &'static str, 1024, BuildHasher::new());
///
/// let value = MY_CACHE.get_or_insert_with(42, |_k| "hi");
/// assert_eq!(value, "hi");
///
/// let new_value = MY_CACHE.get_or_insert_with(42, |_k| "not hi");
/// assert_eq!(new_value, "hi");
/// # }
/// ```
#[macro_export]
macro_rules! static_cache {
    ($K:ty, $V:ty, $size:expr) => {
        $crate::static_cache!($K, $V, $size, Default::default())
    };
    ($K:ty, $V:ty, $size:expr, $hasher:expr) => {{
        static ENTRIES: [$crate::Bucket<($K, $V)>; $size] =
            [const { $crate::Bucket::new() }; $size];
        $crate::Cache::new_static(&ENTRIES, $hasher)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    const fn iters(n: usize) -> usize {
        if cfg!(miri) { n / 10 } else { n }
    }

    type BH = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
    type Cache<K, V> = super::Cache<K, V, BH>;

    struct EpochConfig;
    impl CacheConfig for EpochConfig {
        const EPOCHS: bool = true;
    }
    type EpochCache<K, V> = super::Cache<K, V, BH, EpochConfig>;

    struct NoStatsConfig;
    impl CacheConfig for NoStatsConfig {
        const STATS: bool = false;
    }
    type NoStatsCache<K, V> = super::Cache<K, V, BH, NoStatsConfig>;

    fn new_cache<K: Hash + Eq, V: Clone>(size: usize) -> Cache<K, V> {
        Cache::new(size, Default::default())
    }

    #[test]
    fn basic_get_or_insert() {
        let cache = new_cache(1024);

        let mut computed = false;
        let value = cache.get_or_insert_with(42, |&k| {
            computed = true;
            k * 2
        });
        assert!(computed);
        assert_eq!(value, 84);

        computed = false;
        let value = cache.get_or_insert_with(42, |&k| {
            computed = true;
            k * 2
        });
        assert!(!computed);
        assert_eq!(value, 84);
    }

    #[test]
    fn different_keys() {
        let cache: Cache<String, usize> = new_cache(1024);

        let v1 = cache.get_or_insert_with("hello".to_string(), |s| s.len());
        let v2 = cache.get_or_insert_with("world!".to_string(), |s| s.len());

        assert_eq!(v1, 5);
        assert_eq!(v2, 6);
    }

    #[test]
    fn new_dynamic_allocation() {
        let cache: Cache<u32, u32> = new_cache(64);
        assert_eq!(cache.capacity(), 64);

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));
    }

    #[test]
    fn get_miss() {
        let cache = new_cache::<u64, u64>(64);
        assert_eq!(cache.get(&999), None);
    }

    #[test]
    fn insert_and_get() {
        let cache: Cache<u64, String> = new_cache(64);

        cache.insert(1, "one".to_string());
        cache.insert(2, "two".to_string());
        cache.insert(3, "three".to_string());

        assert_eq!(cache.get(&1), Some("one".to_string()));
        assert_eq!(cache.get(&2), Some("two".to_string()));
        assert_eq!(cache.get(&3), Some("three".to_string()));
        assert_eq!(cache.get(&4), None);
    }

    #[test]
    fn insert_twice() {
        let cache = new_cache(64);

        cache.insert(42, 1);
        assert_eq!(cache.get(&42), Some(1));

        cache.insert(42, 2);
        let v = cache.get(&42);
        assert!(v == Some(1) || v == Some(2));
    }

    #[test]
    fn remove_existing() {
        let cache: Cache<u64, String> = new_cache(64);

        cache.insert(1, "one".to_string());
        assert_eq!(cache.get(&1), Some("one".to_string()));

        let removed = cache.remove(&1);
        assert_eq!(removed, Some("one".to_string()));
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn remove_nonexistent() {
        let cache = new_cache::<u64, u64>(64);
        assert_eq!(cache.remove(&999), None);
    }

    #[test]
    fn get_or_insert_with_ref() {
        let cache: Cache<String, usize> = new_cache(64);

        let key = "hello";
        let value = cache.get_or_insert_with_ref(key, |s| s.len(), |s| s.to_string());
        assert_eq!(value, 5);

        let value2 = cache.get_or_insert_with_ref(key, |_| 999, |s| s.to_string());
        assert_eq!(value2, 5);
    }

    #[test]
    fn get_or_insert_with_ref_different_keys() {
        let cache: Cache<String, usize> = new_cache(1024);

        let v1 = cache.get_or_insert_with_ref("foo", |s| s.len(), |s| s.to_string());
        let v2 = cache.get_or_insert_with_ref("barbaz", |s| s.len(), |s| s.to_string());

        assert_eq!(v1, 3);
        assert_eq!(v2, 6);
    }

    #[test]
    fn capacity() {
        let cache = new_cache::<u64, u64>(256);
        assert_eq!(cache.capacity(), 256);

        let cache2 = new_cache::<u64, u64>(128);
        assert_eq!(cache2.capacity(), 128);
    }

    #[test]
    fn hasher() {
        let cache = new_cache::<u64, u64>(64);
        let _ = cache.hasher();
    }

    #[test]
    fn debug_impl() {
        let cache = new_cache::<u64, u64>(64);
        let debug_str = format!("{:?}", cache);
        assert!(debug_str.contains("Cache"));
    }

    #[test]
    fn bucket_new() {
        let bucket: Bucket<(u64, u64)> = Bucket::new();
        assert_eq!(bucket.tag.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn many_entries() {
        let cache: Cache<u64, u64> = new_cache(1024);
        let n = iters(500);

        for i in 0..n as u64 {
            cache.insert(i, i * 2);
        }

        let mut hits = 0;
        for i in 0..n as u64 {
            if cache.get(&i) == Some(i * 2) {
                hits += 1;
            }
        }
        assert!(hits > 0);
    }

    #[test]
    fn string_keys() {
        let cache: Cache<String, i32> = new_cache(1024);

        cache.insert("alpha".to_string(), 1);
        cache.insert("beta".to_string(), 2);
        cache.insert("gamma".to_string(), 3);

        assert_eq!(cache.get("alpha"), Some(1));
        assert_eq!(cache.get("beta"), Some(2));
        assert_eq!(cache.get("gamma"), Some(3));
    }

    #[test]
    fn zero_values() {
        let cache: Cache<u64, u64> = new_cache(64);

        cache.insert(0, 0);
        assert_eq!(cache.get(&0), Some(0));

        cache.insert(1, 0);
        assert_eq!(cache.get(&1), Some(0));
    }

    #[test]
    fn clone_value() {
        #[derive(Clone, PartialEq, Debug)]
        struct MyValue(u64);

        let cache: Cache<u64, MyValue> = new_cache(64);

        cache.insert(1, MyValue(123));
        let v = cache.get(&1);
        assert_eq!(v, Some(MyValue(123)));
    }

    fn run_concurrent<F>(num_threads: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        thread::scope(|s| {
            for t in 0..num_threads {
                let f = &f;
                s.spawn(move || f(t));
            }
        });
    }

    #[test]
    fn concurrent_reads() {
        let cache: Cache<u64, u64> = new_cache(1024);
        let n = iters(100);

        for i in 0..n as u64 {
            cache.insert(i, i * 10);
        }

        run_concurrent(4, |_| {
            for i in 0..n as u64 {
                let _ = cache.get(&i);
            }
        });
    }

    #[test]
    fn concurrent_writes() {
        let cache: Cache<u64, u64> = new_cache(1024);
        let n = iters(100);

        run_concurrent(4, |t| {
            for i in 0..n {
                cache.insert((t * 1000 + i) as u64, i as u64);
            }
        });
    }

    #[test]
    fn concurrent_read_write() {
        let cache: Cache<u64, u64> = new_cache(256);
        let n = iters(1000);

        run_concurrent(2, |t| {
            for i in 0..n as u64 {
                if t == 0 {
                    cache.insert(i % 100, i);
                } else {
                    let _ = cache.get(&(i % 100));
                }
            }
        });
    }

    #[test]
    fn seqlock_aba() {
        if cfg!(miri) {
            return;
        }

        const VALUE_N: usize = 16;

        let cache: Cache<u64, [u64; VALUE_N]> = new_cache(1024);
        let n = iters(500_000);

        run_concurrent(4, |t| {
            if t == 0 {
                for i in 0..n as u64 {
                    cache.insert(1, [i; VALUE_N]);
                }
            } else {
                for _ in 0..n {
                    if let Some(v) = cache.get(&1) {
                        assert!(v.windows(2).all(|w| w[0] == w[1]), "torn read: {v:?}");
                    }
                }
            }
        });
    }

    #[test]
    fn concurrent_get_or_insert() {
        let cache: Cache<u64, u64> = new_cache(1024);
        let n = iters(100);

        run_concurrent(8, |_| {
            for i in 0..n as u64 {
                let _ = cache.get_or_insert_with(i, |&k| k * 2);
            }
        });

        for i in 0..n as u64 {
            if let Some(v) = cache.get(&i) {
                assert_eq!(v, i * 2);
            }
        }
    }

    #[test]
    #[should_panic = "power of two"]
    fn non_power_of_two() {
        let _ = new_cache::<u64, u64>(100);
    }

    #[test]
    #[should_panic = "len must have its bottom N bits set to zero"]
    fn small_cache() {
        let _ = new_cache::<u64, u64>(2);
    }

    #[test]
    fn power_of_two_sizes() {
        for shift in 2..10 {
            let size = 1 << shift;
            let cache = new_cache::<u64, u64>(size);
            assert_eq!(cache.capacity(), size);
        }
    }

    #[test]
    fn equivalent_key_lookup() {
        let cache: Cache<String, i32> = new_cache(64);

        cache.insert("hello".to_string(), 42);

        assert_eq!(cache.get("hello"), Some(42));
    }

    #[test]
    fn large_values() {
        let cache: Cache<u64, [u8; 1000]> = new_cache(64);

        let large_value = [42u8; 1000];
        cache.insert(1, large_value);

        assert_eq!(cache.get(&1), Some(large_value));
    }

    #[test]
    fn send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<Cache<u64, u64>>();
        assert_sync::<Cache<u64, u64>>();
        assert_send::<Bucket<(u64, u64)>>();
        assert_sync::<Bucket<(u64, u64)>>();
    }

    #[test]
    fn get_or_try_insert_with_ok() {
        let cache = new_cache(1024);

        let mut computed = false;
        let result: Result<u64, &str> = cache.get_or_try_insert_with(42, |&k| {
            computed = true;
            Ok(k * 2)
        });
        assert!(computed);
        assert_eq!(result, Ok(84));

        computed = false;
        let result: Result<u64, &str> = cache.get_or_try_insert_with(42, |&k| {
            computed = true;
            Ok(k * 2)
        });
        assert!(!computed);
        assert_eq!(result, Ok(84));
    }

    #[test]
    fn get_or_try_insert_with_err() {
        let cache: Cache<u64, u64> = new_cache(1024);

        let result: Result<u64, &str> = cache.get_or_try_insert_with(42, |_| Err("failed"));
        assert_eq!(result, Err("failed"));

        assert_eq!(cache.get(&42), None);
    }

    #[test]
    fn get_or_try_insert_with_ref_ok() {
        let cache: Cache<String, usize> = new_cache(64);

        let key = "hello";
        let result: Result<usize, &str> =
            cache.get_or_try_insert_with_ref(key, |s| Ok(s.len()), |s| s.to_string());
        assert_eq!(result, Ok(5));

        let result2: Result<usize, &str> =
            cache.get_or_try_insert_with_ref(key, |_| Ok(999), |s| s.to_string());
        assert_eq!(result2, Ok(5));
    }

    #[test]
    fn get_or_try_insert_with_ref_err() {
        let cache: Cache<String, usize> = new_cache(64);

        let key = "hello";
        let result: Result<usize, &str> =
            cache.get_or_try_insert_with_ref(key, |_| Err("failed"), |s| s.to_string());
        assert_eq!(result, Err("failed"));

        assert_eq!(cache.get(key), None);
    }

    #[test]
    fn drop_on_cache_drop() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone, Hash, Eq, PartialEq)]
        struct DropKey(u64);
        impl Drop for DropKey {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        #[derive(Clone)]
        struct DropValue(#[allow(dead_code)] u64);
        impl Drop for DropValue {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let cache: super::Cache<DropKey, DropValue, BH> =
                super::Cache::new(64, Default::default());
            cache.insert(DropKey(1), DropValue(100));
            cache.insert(DropKey(2), DropValue(200));
            cache.insert(DropKey(3), DropValue(300));
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
        }
        // 3 keys + 3 values = 6 drops
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn drop_on_eviction() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone, Hash, Eq, PartialEq)]
        struct DropKey(u64);
        impl Drop for DropKey {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        #[derive(Clone)]
        struct DropValue(#[allow(dead_code)] u64);
        impl Drop for DropValue {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        DROP_COUNT.store(0, Ordering::SeqCst);
        {
            let cache: super::Cache<DropKey, DropValue, BH> =
                super::Cache::new(64, Default::default());
            cache.insert(DropKey(1), DropValue(100));
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 0);
            // Insert same key again - should evict old entry
            cache.insert(DropKey(1), DropValue(200));
            // Old key + old value dropped = 2
            assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 2);
        }
        // Cache dropped: new key + new value = 2 more
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn epoch_clear() {
        let cache: EpochCache<u64, u64> = EpochCache::new(4096, Default::default());

        assert_eq!(cache.epoch(), 0);

        cache.insert(1, 100);
        cache.insert(2, 200);
        assert_eq!(cache.get(&1), Some(100));
        assert_eq!(cache.get(&2), Some(200));

        cache.clear();
        assert_eq!(cache.epoch(), 1);

        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.get(&2), None);

        cache.insert(1, 101);
        assert_eq!(cache.get(&1), Some(101));

        cache.clear();
        assert_eq!(cache.epoch(), 2);
        assert_eq!(cache.get(&1), None);
    }

    #[test]
    fn epoch_wrap_around() {
        let cache: EpochCache<u64, u64> = EpochCache::new(4096, Default::default());

        for _ in 0..300 {
            cache.insert(42, 123);
            assert_eq!(cache.get(&42), Some(123));
            cache.clear();
            assert_eq!(cache.get(&42), None);
        }
    }

    #[test]
    fn no_stats_config() {
        let cache: NoStatsCache<u64, u64> = NoStatsCache::new(64, Default::default());

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));
        assert_eq!(cache.get(&999), None);

        cache.insert(1, 200);
        assert_eq!(cache.get(&1), Some(200));

        cache.remove(&1);
        assert_eq!(cache.get(&1), None);

        let v = cache.get_or_insert_with(42, |&k| k * 2);
        assert_eq!(v, 84);
    }

    #[test]
    fn epoch_wraparound_stays_cleared() {
        let cache: EpochCache<u64, u64> = EpochCache::new(4096, Default::default());

        cache.insert(42, 123);
        assert_eq!(cache.get(&42), Some(123));

        for i in 0..2048 {
            cache.clear();
            assert_eq!(cache.get(&42), None, "failed at clear #{i}");
        }
    }

    #[test]
    fn remove_seqlock_type() {
        let cache = new_cache::<u64, u64>(64);

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));

        let removed = cache.remove(&1);
        assert_eq!(removed, Some(100));
        assert_eq!(cache.get(&1), None);

        cache.insert(1, 200);
        assert_eq!(cache.get(&1), Some(200));
    }

    #[test]
    fn remove_then_reinsert_seqlock() {
        let cache = new_cache::<u64, u64>(64);

        for i in 0..100u64 {
            cache.insert(1, i);
            assert_eq!(cache.get(&1), Some(i));
            assert_eq!(cache.remove(&1), Some(i));
            assert_eq!(cache.get(&1), None);
        }
    }

    #[test]
    fn epoch_with_needs_drop() {
        let cache: EpochCache<String, String> = EpochCache::new(4096, Default::default());

        cache.insert("key".to_string(), "value".to_string());
        assert_eq!(cache.get("key"), Some("value".to_string()));

        cache.clear();
        assert_eq!(cache.get("key"), None);

        cache.insert("key".to_string(), "value2".to_string());
        assert_eq!(cache.get("key"), Some("value2".to_string()));
    }

    #[test]
    fn epoch_remove() {
        let cache: EpochCache<u64, u64> = EpochCache::new(4096, Default::default());

        cache.insert(1, 100);
        assert_eq!(cache.remove(&1), Some(100));
        assert_eq!(cache.get(&1), None);

        cache.insert(1, 200);
        assert_eq!(cache.get(&1), Some(200));

        cache.clear();
        assert_eq!(cache.get(&1), None);
        assert_eq!(cache.remove(&1), None);
    }

    #[test]
    fn no_stats_needs_drop() {
        let cache: NoStatsCache<String, String> = NoStatsCache::new(64, Default::default());

        cache.insert("a".to_string(), "b".to_string());
        assert_eq!(cache.get("a"), Some("b".to_string()));

        cache.insert("a".to_string(), "c".to_string());
        assert_eq!(cache.get("a"), Some("c".to_string()));

        cache.remove(&"a".to_string());
        assert_eq!(cache.get("a"), None);
    }

    #[test]
    fn no_stats_get_or_insert() {
        let cache: NoStatsCache<String, usize> = NoStatsCache::new(64, Default::default());

        let v = cache.get_or_insert_with_ref("hello", |s| s.len(), |s| s.to_string());
        assert_eq!(v, 5);

        let v2 = cache.get_or_insert_with_ref("hello", |_| 999, |s| s.to_string());
        assert_eq!(v2, 5);
    }

    #[test]
    fn epoch_concurrent_seqlock() {
        if cfg!(miri) {
            return;
        }

        let cache: EpochCache<u64, u64> = EpochCache::new(4096, Default::default());
        let n = iters(10_000);

        run_concurrent(4, |t| {
            for i in 0..n as u64 {
                match t {
                    0 => {
                        cache.insert(i % 50, i);
                    }
                    1 => {
                        let _ = cache.get(&(i % 50));
                    }
                    2 => {
                        if i % 100 == 0 {
                            cache.clear();
                        }
                    }
                    _ => {
                        let _ = cache.remove(&(i % 50));
                    }
                }
            }
        });
    }
}
