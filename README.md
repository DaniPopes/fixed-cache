# fixed-cache

A minimalistic, lock-free, fixed-size cache for Rust.

This crate provides a concurrent set-associative cache with a fixed number of entries.
It is designed for high-performance scenarios where you need fast, thread-safe caching
with predictable memory usage and minimal overhead.

## Features

- **Fixed size**: Memory is allocated once at creation time
- **Lock-free reads**: Uses atomic operations for thread-safe access without blocking
- **Zero dependencies** (optional `rapidhash` for faster hashing)
- **`no_std` compatible** (with `alloc`)
- **Static initialization**: Create caches at compile time with the `static_cache!` macro

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
fixed-cache = "0.1"
```

### Basic Example

```rust
use fixed_cache::Cache;

// Create a cache with 1024 entries
let cache: Cache<u64, u64> = Cache::new(1024, Default::default());

// Insert and retrieve values
cache.insert(42, 100);
assert_eq!(cache.get(&42), Some(100));

// Use get_or_insert_with for lazy initialization
let value = cache.get_or_insert_with(123, |&k| k * 2);
assert_eq!(value, 246);
```

### Static Cache

For global caches that need to be initialized at compile time:

```rust,ignore
use fixed_cache::{Cache, static_cache};

// Requires a const-compatible hasher (e.g., rapidhash with the `rapidhash` feature)
type MyBuildHasher = todo!();

static CACHE: Cache<u64, u64, MyBuildHasher> = static_cache!(u64, u64, 1024, MyBuildHasher::new());

fn lookup(key: u64) -> u64 {
    CACHE.get_or_insert_with(key, |&k| k * 2) // your computation here
}
```

## How It Works

The cache uses a set-associative design where each key maps to exactly one bucket
based on its hash. When a collision occurs (two keys hash to the same bucket),
the new value evicts the old one. This means:

- **O(1) lookups and insertions** - no linked lists or trees to traverse
- **No resizing** - memory usage is fixed and predictable
- **Possible evictions** - entries may be evicted even if the cache isn't "full"

This design is ideal for memoization caches where:
- Cache misses are acceptable (you can recompute the value)
- Predictable latency is more important than perfect hit rates
- Memory bounds must be strictly controlled

## Limitations

- **No iteration**: You cannot iterate over cached entries
- **No removal**: Individual entries cannot be explicitly removed
- **Eviction on collision**: Hash collisions cause immediate eviction

## Feature Flags

- `rapidhash` - Use [rapidhash](https://crates.io/crates/rapidhash) for faster hashing (recommended)
- `stats` - Enable statistics tracking (hits, misses, collisions) via custom handlers
- `nightly` - Enable nightly-only optimizations

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
