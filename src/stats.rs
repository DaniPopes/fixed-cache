use std::{
    any::TypeId,
    fmt,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// A type-erased reference that can be downcast even for non-`'static` types.
///
/// Unlike `&dyn Any`, this works with non-`'static` types by storing the `TypeId`
/// separately using the `typeid` crate which can compute `TypeId` for any type.
#[derive(Clone, Copy)]
pub struct AnyRef<'a> {
    ptr: *const (),
    type_id: TypeId,
    _marker: PhantomData<&'a ()>,
}

impl<'a> AnyRef<'a> {
    /// Creates a new `AnyRef` from a reference.
    #[inline]
    pub(crate) fn new<T>(value: &'a T) -> Self {
        Self {
            ptr: value as *const T as *const (),
            type_id: typeid::of::<T>(),
            _marker: PhantomData,
        }
    }

    /// Returns a raw pointer to the referenced value.
    #[inline]
    pub fn as_ptr(&self) -> *const () {
        self.ptr
    }

    /// Returns the `TypeId` of the referenced value.
    #[inline]
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Attempts to downcast to a concrete type `T`.
    ///
    /// Returns `Some(&T)` if the type matches, `None` otherwise.
    #[inline]
    pub fn downcast_ref<T>(&self) -> Option<&'a T> {
        if self.type_id == typeid::of::<T>() {
            // SAFETY: We verified the type matches and the lifetime is preserved.
            Some(unsafe { &*(self.ptr as *const T) })
        } else {
            None
        }
    }
}

impl fmt::Debug for AnyRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyRef").field("type_id", &self.type_id).finish_non_exhaustive()
    }
}

/// Trait for handling cache statistics events.
///
/// Implement this trait to provide custom handling for cache hits, misses, and collisions.
/// All methods have default implementations that do nothing, allowing you to override only
/// the events you care about.
pub trait StatsHandler<K, V>: Send + Sync {
    /// Called when a cache hit occurs (key found and value returned).
    fn on_hit(&self, key: &K, value: &V) {
        let _ = key;
        let _ = value;
    }

    /// Called when a cache miss occurs (key not found).
    ///
    /// The `key` parameter is a type-erased reference to the lookup key (`Q`), which may be a
    /// different type than `K`.
    fn on_miss(&self, key: AnyRef<'_>) {
        let _ = key;
    }

    /// Called when a collision occurs (same bucket, different key - entry will be evicted).
    ///
    /// Note that `on_miss` is also called after a collision.
    ///
    /// The `key` parameter is a type-erased reference to the lookup key (`Q`), which may be a
    /// different type than `K`.
    fn on_collision(&self, new_key: AnyRef<'_>, existing_key: &K, existing_value: &V) {
        let _ = new_key;
        let _ = existing_key;
        let _ = existing_value;
    }
}

/// Default statistics handler that tracks counts using atomic integers.
///
/// This is a simple implementation that just counts the number of hits, misses, and collisions.
pub struct CountingStatsHandler {
    hits: AtomicU64,
    misses: AtomicU64,
    collisions: AtomicU64,
}

impl CountingStatsHandler {
    /// Creates a new counting stats handler with all counters initialized to zero.
    pub const fn new() -> Self {
        Self { hits: AtomicU64::new(0), misses: AtomicU64::new(0), collisions: AtomicU64::new(0) }
    }

    /// Returns the number of cache hits.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Returns the number of cache misses.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Returns the number of collisions.
    pub fn collisions(&self) -> u64 {
        self.collisions.load(Ordering::Relaxed)
    }

    /// Resets all counters to zero.
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.collisions.store(0, Ordering::Relaxed);
    }
}

impl Default for CountingStatsHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> StatsHandler<K, V> for CountingStatsHandler {
    fn on_hit(&self, _key: &K, _value: &V) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    fn on_miss(&self, _key: AnyRef<'_>) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    fn on_collision(&self, _new_key: AnyRef<'_>, _existing_key: &K, _existing_value: &V) {
        self.collisions.fetch_add(1, Ordering::Relaxed);
    }
}

impl<K, V, T: StatsHandler<K, V>> StatsHandler<K, V> for Arc<T> {
    #[inline]
    fn on_hit(&self, key: &K, value: &V) {
        (**self).on_hit(key, value);
    }

    #[inline]
    fn on_miss(&self, key: AnyRef<'_>) {
        (**self).on_miss(key);
    }

    #[inline]
    fn on_collision(&self, new_key: AnyRef<'_>, existing_key: &K, existing_value: &V) {
        (**self).on_collision(new_key, existing_key, existing_value);
    }
}

/// Wrapper around a dynamic stats handler.
///
/// This struct is stored in the cache when statistics tracking is enabled.
pub struct Stats<K, V> {
    handler: Box<dyn StatsHandler<K, V>>,
}

impl<K, V> Stats<K, V> {
    /// Creates a new stats wrapper with a boxed handler.
    #[inline]
    pub fn new<H: StatsHandler<K, V> + 'static>(handler: H) -> Self {
        Self { handler: Box::new(handler) }
    }

    /// Returns a reference to the underlying handler.
    #[inline]
    pub fn handler(&self) -> &dyn StatsHandler<K, V> {
        &*self.handler
    }

    #[inline]
    pub(crate) fn record_hit(&self, key: &K, value: &V) {
        self.handler.on_hit(key, value);
    }

    #[inline]
    pub(crate) fn record_miss(&self, key: AnyRef<'_>) {
        self.handler.on_miss(key);
    }

    #[inline]
    pub(crate) fn record_collision(
        &self,
        new_key: AnyRef<'_>,
        existing_key: &K,
        existing_value: &V,
    ) {
        self.handler.on_collision(new_key, existing_key, existing_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cache;
    use std::sync::Arc;

    type BH = std::hash::BuildHasherDefault<rapidhash::fast::RapidHasher<'static>>;

    #[test]
    fn counting_stats_handler_basic() {
        let handler = CountingStatsHandler::new();
        assert_eq!(handler.hits(), 0);
        assert_eq!(handler.misses(), 0);
        assert_eq!(handler.collisions(), 0);

        StatsHandler::<u64, u64>::on_hit(&handler, &1, &2);
        assert_eq!(handler.hits(), 1);

        StatsHandler::<u64, u64>::on_miss(&handler, AnyRef::new(&1u64));
        assert_eq!(handler.misses(), 1);

        StatsHandler::<u64, u64>::on_collision(&handler, AnyRef::new(&1u64), &1, &2);
        assert_eq!(handler.collisions(), 1);

        handler.reset();
        assert_eq!(handler.hits(), 0);
        assert_eq!(handler.misses(), 0);
        assert_eq!(handler.collisions(), 0);
    }

    #[test]
    fn cache_with_stats_hits_and_misses() {
        let handler = Arc::new(CountingStatsHandler::new());
        let stats = Stats::new(Arc::clone(&handler));
        let cache: Cache<u64, u64, BH> = Cache::new(64, Default::default()).with_stats(Some(stats));

        assert_eq!(cache.get(&42), None);
        assert_eq!(handler.misses(), 1);

        cache.insert(42, 100);

        assert_eq!(cache.get(&42), Some(100));
        assert_eq!(handler.hits(), 1);
        assert_eq!(handler.misses(), 1);

        assert_eq!(cache.get(&99), None);
        assert_eq!(handler.misses(), 2);
    }

    #[test]
    fn cache_with_stats_collisions_on_get() {
        use std::hash::{Hash, Hasher};

        #[derive(Clone, Eq, PartialEq)]
        struct CollidingKey(u64, u64);

        impl Hash for CollidingKey {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.0.hash(state);
            }
        }

        let handler = Arc::new(CountingStatsHandler::new());
        let stats = Stats::new(Arc::clone(&handler));
        let cache: Cache<CollidingKey, u64, BH> =
            Cache::new(64, Default::default()).with_stats(Some(stats));

        cache.insert(CollidingKey(1, 1), 100);
        let result = cache.get(&CollidingKey(1, 2));
        assert!(result.is_none());

        assert_eq!(handler.collisions(), 1);
    }

    #[test]
    fn get_or_insert_with_stats() {
        let handler = Arc::new(CountingStatsHandler::new());
        let stats = Stats::new(Arc::clone(&handler));
        let cache: Cache<u64, u64, BH> = Cache::new(64, Default::default()).with_stats(Some(stats));

        let value = cache.get_or_insert_with(42, |&k| k * 2);
        assert_eq!(value, 84);
        assert_eq!(handler.misses(), 1);

        let value = cache.get_or_insert_with(42, |&k| k * 3);
        assert_eq!(value, 84);
        assert_eq!(handler.hits(), 1);
    }

    #[test]
    fn boxed_stats_handler() {
        let handler = Arc::new(CountingStatsHandler::new());
        let stats = Stats::new(Arc::clone(&handler));
        let cache: Cache<u64, u64, BH> = Cache::new(64, Default::default()).with_stats(Some(stats));

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));
        assert_eq!(cache.get(&2), None);
        assert_eq!(handler.hits(), 1);
        assert_eq!(handler.misses(), 1);
    }

    #[test]
    fn cache_without_stats() {
        let cache: Cache<u64, u64, BH> = Cache::new(64, Default::default());

        cache.insert(1, 100);
        assert_eq!(cache.get(&1), Some(100));
        assert!(cache.stats().is_none());
    }

    #[test]
    fn anyref_downcast_in_handler() {
        use std::sync::Mutex;

        struct CapturingHandler {
            missed_keys: Mutex<Vec<String>>,
        }

        impl CapturingHandler {
            fn new() -> Self {
                Self { missed_keys: Mutex::new(Vec::new()) }
            }
        }

        impl StatsHandler<String, u64> for CapturingHandler {
            fn on_miss(&self, key: AnyRef<'_>) {
                if let Some(k) = key.downcast_ref::<&str>() {
                    self.missed_keys.lock().unwrap().push((*k).to_string());
                } else if let Some(k) = key.downcast_ref::<&String>() {
                    self.missed_keys.lock().unwrap().push((*k).clone());
                }
            }
        }

        let handler = Arc::new(CapturingHandler::new());
        let stats = Stats::new(Arc::clone(&handler));
        let cache: Cache<String, u64, BH> =
            Cache::new(64, Default::default()).with_stats(Some(stats));

        assert_eq!(cache.get("hello"), None);
        assert_eq!(cache.get("world"), None);
        assert_eq!(cache.get(&"foo".to_string()), None);

        assert_eq!(*handler.missed_keys.lock().unwrap(), vec!["hello", "world", "foo"]);
    }
}
