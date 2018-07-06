/* 
 * lru_cache.h
 *
 * Header-only implementation of a least-recently-used cache in C++.
 * Requires C++14 or above.
 *
 * MIT License
 * 
 * Copyright (c) 2018 Gregory Maurice Green
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#ifndef _LRU_CACHE_H__
#define _LRU_CACHE_H__


#include <cstdint>
#include <list>
#include <unordered_map>
#include <functional>
#include <vector>


#define LRUCACHE_VERBOSE 1 // Set to 1 for hit/miss stats, 0 for quiet

#if LRUCACHE_VERBOSE
#include <iostream>
#endif


namespace LRUCache {


template<class T>
class VectorHasher {
    // Hashing function for vectors, required to use them as
    // keys in an unordered_map. Adapted from HolKann's
    // StackOverflow answer: <https://stackoverflow.com/a/27216842/1103939>.
public:
    std::size_t operator()(const std::vector<T>& vec) const;
};


template<class T>
std::size_t VectorHasher<T>::operator()(
        const std::vector<T>& vec) const
{
    std::size_t seed = vec.size();
    for(auto& v : vec) {
        seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}



template<class TKey, class TValue, class THash=std::hash<TKey>>
class LRUCache {
public:
    LRUCache(uint32_t capacity, const TValue& empty_value);
    ~LRUCache();

    TValue get(const TKey& key);
    void set(const TKey& key, const TValue& value);

protected:
    uint32_t capacity; // Max # of items to store.
    TValue empty_value; // Value to return when item not in cache

    // Order in which keys have been accessed. Most recent first.
    std::list<TKey> access_order;

    // Map to store (key, value) pairs.
    std::unordered_map<TKey, TValue, THash> cache;

    // Map for quick lookup of access order.
    std::unordered_map<TKey, typename std::list<TKey>::iterator, THash> access_order_it;

    typename std::unordered_map<TKey, TValue, THash>::iterator push_front(const TKey& key, const TValue& value);
    void bring_to_front(const TKey& key);
    void remove_lru();

    #if LRUCACHE_VERBOSE
    uint64_t n_hit, n_miss, n_replace;
    #endif
};


template<class TKey, class TValue, class THash=std::hash<TKey>>
class CachedFunction : public LRUCache<TKey,TValue,THash> {
public:
    CachedFunction(std::function<TValue(const TKey&)> f, uint32_t capacity);
    CachedFunction(std::function<TValue(const TKey&)> f, uint32_t capacity, const TValue& empty_value);
    ~CachedFunction();
    
    TValue eval(const TKey& arg);
    TValue operator()(const TKey& arg);
    
    void eval(const TKey& arg, std::function<void(TValue&)>);
    void operator()(const TKey& arg, std::function<void(TValue&)>);

    TValue& eval_ref(const TKey& arg);
    
private:
    std::function<TValue(const TKey&)> f;
};


template<class TKey, class TValue, class THash>
CachedFunction<TKey, TValue, THash>::CachedFunction(std::function<TValue(const TKey&)> f, uint32_t capacity)
    : LRUCache<TKey, TValue, THash>(capacity, TValue()), f(f)
{}


template<class TKey, class TValue, class THash>
CachedFunction<TKey, TValue, THash>::CachedFunction(std::function<TValue(const TKey&)> f, uint32_t capacity, const TValue& empty_value)
    : LRUCache<TKey, TValue, THash>(capacity, empty_value), f(f)
{}


template<class TKey, class TValue, class THash>
CachedFunction<TKey, TValue, THash>::~CachedFunction() {}


template<class TKey, class TValue, class THash>
TValue CachedFunction<TKey, TValue, THash>::eval(const TKey& arg) {
    // Look up key in cache
    auto cache_it = this->cache.find(arg);
    if(cache_it == this->cache.end()) { // Key not found
        TValue value = f(arg);
        this->push_front(arg, value);
        return value;
    } else { // Key found
        #if LRUCACHE_VERBOSE
        this->n_hit++;
        #endif

        // Update access order
        this->bring_to_front(arg);
        // Return cached value
        return cache_it->second;
    }
}


template<class TKey, class TValue, class THash>
void CachedFunction<TKey, TValue, THash>::eval(const TKey& arg, std::function<void(TValue&)> g) {
    // Look up key in cache
    auto cache_it = this->cache.find(arg);
    if(cache_it == this->cache.end()) { // Key not found
        TValue value = f(arg);
        this->push_front(arg, value);
        // Operate on newly computed value
        g(value);
    } else { // Key found
        #if LRUCACHE_VERBOSE
        this->n_hit++;
        #endif

        // Update access order
        this->bring_to_front(arg);
        // Operate on cached value
        g(cache_it->second);
    }
}


template<class TKey, class TValue, class THash>
TValue& CachedFunction<TKey, TValue, THash>::eval_ref(const TKey& arg) {
    // Look up key in cache
    auto cache_it = this->cache.find(arg);
    if(cache_it == this->cache.end()) { // Key not found
        TValue value = f(arg);
        auto it = this->push_front(arg, value);
        return it->second;
    } else { // Key found
        #if LRUCACHE_VERBOSE
        this->n_hit++;
        #endif

        // Update access order
        this->bring_to_front(arg);
        // Return cached value
        return cache_it->second;
    }
}


template<class TKey, class TValue, class THash>
TValue CachedFunction<TKey, TValue, THash>::operator()(const TKey& arg) {
    return eval(arg);
}


template<class TKey, class TValue, class THash>
void CachedFunction<TKey, TValue, THash>::operator()(const TKey& arg, std::function<void(TValue&)> g) {
    return eval(arg, g);
}


template<class TKey, class TValue, class THash>
LRUCache<TKey, TValue, THash>::LRUCache(uint32_t capacity, const TValue& empty_value)
    : capacity(capacity), empty_value(empty_value)
{
    #if LRUCACHE_VERBOSE
    n_hit = 0;
    n_miss = 0;
    n_replace = 0;
    #endif
}


template<class TKey, class TValue, class THash>
LRUCache<TKey, TValue, THash>::~LRUCache() {
    #if LRUCACHE_VERBOSE
    std::cout << "LRUCache hits/misses/replacements = "
              << n_hit << " / "
              << n_miss << " / "
              << n_replace << std::endl;
    #endif
}


template<class TKey, class TValue, class THash>
TValue LRUCache<TKey, TValue, THash>::get(const TKey& key) {
    // Look up key in cache
    auto cache_it = cache.find(key);
    if(cache_it == cache.end()) { // Key not found
        #if LRUCACHE_VERBOSE
        n_miss++;
        #endif

        return empty_value;
    } else { // Key found
        #if LRUCACHE_VERBOSE
        n_hit++;
        #endif

        // Update access order
        bring_to_front(key);
        // Return cached value
        return cache_it->second;
    }
}


template<class TKey, class TValue, class THash>
typename std::unordered_map<TKey, TValue, THash>::iterator LRUCache<TKey, TValue, THash>::push_front(
        const TKey& key,
        const TValue& value)
{
    #if LRUCACHE_VERBOSE
    n_miss++;
    #endif

    // Add (key, value) pair to cache
    auto cache_it = cache.insert(std::make_pair(key, value)).first;
    // Insert key into access-order list
    access_order.push_front(key);
    // Update access-order lookup
    access_order_it.insert(std::make_pair(key, access_order.begin()));
    // If over capacity, remove least-recently-used key
    if(cache.size() > capacity) {
        remove_lru();
    }

    return cache_it;
}

template<class TKey, class TValue, class THash>
void LRUCache<TKey, TValue, THash>::set(const TKey& key, const TValue& value) {
    // Look up key in cache
    auto cache_it = cache.find(key);
    if(cache_it == cache.end()) { // Key not in cache
        push_front(key, value);
    } else { // Key found
        #if LRUCACHE_VERBOSE
        n_hit++;
        #endif

        // Bring key to front of access order list
        bring_to_front(key);
        // Update value
        cache_it->second = value;
    }
}


template<class TKey, class TValue, class THash>
void LRUCache<TKey, TValue, THash>::remove_lru() {
    #if LRUCACHE_VERBOSE
    n_replace++;
    #endif

    // Look up key of least-recently-used element
    const TKey& key = access_order.back();
    // Erase key from cache and access-order-iterator lookup
    cache.erase(key);
    access_order_it.erase(key);
    // Erase key from access-order list
    access_order.pop_back();
}


template<class TKey, class TValue, class THash>
void LRUCache<TKey, TValue, THash>::bring_to_front(const TKey& key) {
    auto ao_it = access_order_it.find(key); // Assumed to be found
    if(ao_it->second != access_order.begin()) {
        // Move key to front of access-order list
        access_order.erase(ao_it->second);
        access_order.push_front(key);
        // Update location of key in access-order-iterator lookup
        ao_it->second = access_order.begin();
    }
}


} // namespace LRUCache


#endif // _LRU_CACHE_H__
