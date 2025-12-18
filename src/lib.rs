#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::new_without_default)]

use equivalent::Equivalent;
use std::{
    cell::UnsafeCell,
    convert::Infallible,
    fmt,
    hash::{BuildHasher, Hash},
    mem::MaybeUninit,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "stats")]
mod stats;
#[cfg(feature = "stats")]
#[cfg_attr(docsrs, doc(cfg(feature = "stats")))]
pub use stats::{AnyRef, CountingStatsHandler, Stats, StatsHandler};

const NEEDED_BITS: usize = 2;
const LOCKED_BIT: usize = 1 << 0;
const ALIVE_BIT: usize = 1 << 1;

#[cfg(feature = "rapidhash")]
type DefaultBuildHasher = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;
#[cfg(not(feature = "rapidhash"))]
type DefaultBuildHasher = std::hash::RandomState;

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
/// - **No iteration or removal**: Individual entries cannot be enumerated or explicitly removed.
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
pub struct Cache<K, V, S = DefaultBuildHasher> {
    entries: *const [Bucket<(K, V)>],
    build_hasher: S,
    #[cfg(feature = "stats")]
    stats: Option<Stats<K, V>>,
    drop: bool,
}

impl<K, V, S> fmt::Debug for Cache<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cache").finish_non_exhaustive()
    }
}

// SAFETY: `Cache` is safe to share across threads because `Bucket` uses atomic operations.
unsafe impl<K: Send, V: Send, S: Send> Send for Cache<K, V, S> {}
unsafe impl<K: Send, V: Send, S: Sync> Sync for Cache<K, V, S> {}

impl<K, V, S> Cache<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
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
        let entries =
            Box::into_raw((0..num).map(|_| Bucket::new()).collect::<Vec<_>>().into_boxed_slice());
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
    #[cfg(feature = "stats")]
    #[inline]
    pub fn with_stats(mut self, stats: Option<Stats<K, V>>) -> Self {
        self.set_stats(stats);
        self
    }

    /// Sets the cache's statistics.
    #[cfg(feature = "stats")]
    #[inline]
    pub fn set_stats(&mut self, stats: Option<Stats<K, V>>) {
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
        }
    }

    #[inline]
    const fn len_assertion(len: usize) {
        // We need `NEEDED_BITS` bits to store metadata for each entry.
        // Since we calculate the tag mask based on the index mask, and the index mask is (len - 1),
        // we assert that the length's bottom `NEEDED_BITS` bits are zero.
        assert!(len.is_power_of_two(), "length must be a power of two");
        assert!(
            (len & ((1 << NEEDED_BITS) - 1)) == 0,
            "len must have its bottom N bits set to zero"
        );
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

impl<K, V, S> Cache<K, V, S>
where
    K: Hash + Eq,
    V: Clone,
    S: BuildHasher,
{
    const NEEDS_DROP: bool = Bucket::<(K, V)>::NEEDS_DROP;

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
        if bucket.try_lock(Some(tag)) {
            // SAFETY: We hold the lock and bucket is alive, so we have exclusive access.
            let (ck, v) = unsafe { (*bucket.data.get()).assume_init_ref() };
            if key.equivalent(ck) {
                #[cfg(feature = "stats")]
                if let Some(stats) = &self.stats {
                    stats.record_hit(ck, v);
                }
                let v = v.clone();
                bucket.unlock(tag);
                return Some(v);
            }
            #[cfg(feature = "stats")]
            if let Some(stats) = &self.stats {
                stats.record_collision(AnyRef::new(&key), ck, v);
            }
            bucket.unlock(tag);
        }
        #[cfg(feature = "stats")]
        if let Some(stats) = &self.stats {
            stats.record_miss(AnyRef::new(&key));
        }
        None
    }

    /// Insert an entry into the cache.
    pub fn insert(&self, key: K, value: V) {
        let (bucket, tag) = self.calc(&key);
        self.insert_inner(|| key, || value, bucket, tag);
    }

    #[inline]
    fn insert_inner(
        &self,
        make_key: impl FnOnce() -> K,
        make_value: impl FnOnce() -> V,
        bucket: &Bucket<(K, V)>,
        tag: usize,
    ) {
        if let Some(prev_tag) = bucket.try_lock_ret(None) {
            // SAFETY: We hold the lock, so we have exclusive access.
            unsafe {
                let data = (&mut *bucket.data.get()).as_mut_ptr();
                // Drop old value if bucket was alive.
                if Self::NEEDS_DROP && (prev_tag & ALIVE_BIT) != 0 {
                    std::ptr::drop_in_place(data);
                }
                (&raw mut (*data).0).write(make_key());
                (&raw mut (*data).1).write(make_value());
            }
            bucket.unlock(tag);
        }
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
        self.get_or_try_insert_with(key, |key| Ok::<_, Infallible>(f(key))).unwrap()
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
        self.get_or_try_insert_with_ref(key, |key| Ok::<_, Infallible>(f(key)), cvt).unwrap()
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
        self.insert_inner(|| cvt(key), || value.clone(), bucket, tag);
        Ok(value)
    }

    #[inline]
    fn calc<Q: ?Sized + Hash + Equivalent<K>>(&self, key: &Q) -> (&Bucket<(K, V)>, usize) {
        let hash = self.hash_key(key);
        // SAFETY: index is masked to be within bounds.
        let bucket = unsafe { (&*self.entries).get_unchecked(hash & self.index_mask()) };
        let mut tag = hash & self.tag_mask();
        if Self::NEEDS_DROP {
            tag |= ALIVE_BIT;
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

impl<K, V, S> Drop for Cache<K, V, S> {
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
    fn try_lock(&self, expected: Option<usize>) -> bool {
        self.try_lock_ret(expected).is_some()
    }

    #[inline]
    fn try_lock_ret(&self, expected: Option<usize>) -> Option<usize> {
        let state = self.tag.load(Ordering::Relaxed);
        if let Some(expected) = expected {
            if state != expected {
                return None;
            }
        } else if state & LOCKED_BIT != 0 {
            return None;
        }
        self.tag
            .compare_exchange(state, state | LOCKED_BIT, Ordering::Acquire, Ordering::Relaxed)
            .ok()
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
}
