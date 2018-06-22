
#ifndef __BRIDGING_SAMPLER_H__
#define __BRIDGING_SAMPLER_H__

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>

#include <iostream>


namespace bridgesamp {


class VectorHasher {
    // Hashing function for vectors, required to use them as
    // keys in an unordered_map. Taken from HolKann's
    // StackOverflow answer: <https://stackoverflow.com/a/27216842/1103939>.
public:
    std::size_t operator()(const std::vector<uint16_t>& vec) const;
};


struct Node {
    // An individual state in the combinatorial space. Only need
    // to store p and log(p), since the nodes will be stored in a
    // map, with the state as the key.
    double p, logp;
};


// An unordered_map that maps state vectors -> (p, log(p))
typedef std::unordered_map<std::vector<uint16_t>, Node, VectorHasher> NodeMap;


class BridgingSampler {
private:
    uint16_t n_dim;     // # of integers to define a state
    uint16_t n_samples; // # of samples per dimension
    
    NodeMap node; // All visited nodes
    NodeMap::iterator state; // Points to current node
    uint16_t state_rank; // Rank of current state
    
    // Function that returns log(p) of node
    std::function<double(const std::vector<uint16_t>&)> eval_node;

    // Function that returns log(p) of several nodes that differ only in
    // one dimension
    std::function<double(
            uint16_t, // dimension that nodes differ in
            const std::vector<uint16_t>&, // sample # in each dimension
            std::vector<double>&  // holds log(p) of each state
    )> eval_conditional;

    double p0, logp0; // Assumed value of p (or log(p)) for unexplored nodes
    double log_n_samples;

    // Sampling parameters
    std::vector<double> b_prob; // Probability of backward transition at each rank
    std::vector<double> f_prob; // Probability of forward transition at each rank

    // Random numbers
    std::mt19937 r;
    std::uniform_int_distribution<uint16_t> r_dim;    // Draw a random dimension
    std::uniform_int_distribution<uint16_t> r_samp;   // Draw a random sample
    std::uniform_real_distribution<double> r_uniform; // Draw from U(0,1)

    // Workspaces
    //   - Worspaces ending in _dim_ws have capacity equal to n_dim
    //   -     "       "    "  _samp_ws "      "       "   "  n_samples
    //   Each workspace is named according to the member function that uses it
    std::vector<uint16_t> _rand_state_dim_ws;
    std::vector<uint16_t> _gibbs_idx_samp_ws;
    std::vector<double> _gibbs_lnp_samp_ws;
    std::vector<uint16_t> _gibbs_state_dim_ws;
    std::vector<uint16_t> _transition_state_dim_ws;
    std::vector<uint16_t> _percolate_dim_ws;

    // Navigation
    //std::map<std::vector<uint16_t>, Node>::iterator up(
    //        std::map<std::vector<uint16_t>, Node>::iterator s,
    //        int
    
    // Returns an iterator to the node specified by the given sample numbers
    NodeMap::iterator get_node(const std::vector<uint16_t>& samp);

    // # of children of node of given order
    double n_children(uint16_t order);
    
    // # of dimensions that are empty in key
    uint16_t get_n_empty(const std::vector<uint16_t>& samp);

    // Sampling routines
    void percolate_up(const NodeMap::iterator& n);
    
    void _lazy_gibbs_inner(std::vector<uint16_t>& starting_state,
                           uint16_t dim);

    // Get dimensions which are empty
    void get_empty_dims(const NodeMap::iterator& n,
                        std::vector<uint16_t>& dims_out);
    
    // Get dimensions which are set (non-empty)
    void get_nonempty_dims(const NodeMap::iterator& n,
                           std::vector<uint16_t>& dims_out);

    double get_lnp0_of_rank(uint16_t rank);

public:
    BridgingSampler(uint16_t n_dim,
                    uint16_t n_samples,
                    std::function<double(const std::vector<uint16_t>&)> logp_node);
    
    void step(); // Choose and execute a step type

    void randomize_state(); // Jump to randomly chosen base state
    
    // Take a Gibbs step in the specified dimension
    void gibbs(uint16_t dim);

    // Take a Gibbs step in the specified dimension, without evaluating
    // unexplored states.
    void lazy_gibbs(uint16_t dim);
    
    // Take a Gibbs step, choosing the dimension randomly
    void lazy_gibbs_choose_dim();
    
    // Transition to a randomly selected state one level up in the hierarchy
    void transition_backward();
    
    // Transition to a randomly selected state one level down in the hierarchy
    void transition_forward();

    // Set conditional probability function, to be used in Gibbs steps
    void set_conditional_prob(
        std::function<double(
            uint16_t, // dimension that nodes differ in
            const std::vector<uint16_t>&, // sample # in each dimension
            std::vector<double>&  // holds log(p) of each state
        )> f
    );

    void set_logp0(double _logp0);

    // Statistics
    double fill_factor() const; // Fraction of nodes explored

    // Getters
    NodeMap::const_iterator cbegin() const;
    NodeMap::const_iterator cend() const;

    uint16_t get_n_dim() const;
    uint16_t get_n_samples() const;

    const std::vector<uint16_t>& get_state() const;
    double get_logp() const;
    uint16_t get_state_rank() const; // Get current level in hierarchy
};


class CombinationGenerator {
    // Cycles through combinations of elements. Generates
    // n choose r combinations.
public:
    // n choose r
    CombinationGenerator(unsigned int n, unsigned int r);

    // Set <out_idx> to the next combination. Returns false
    // if the final combination has already been reached.
    bool next(std::vector<unsigned int>& out_idx);

    // Reset the combination generator to the beginning
    void reset();

private:
    std::vector<bool> mask; // True for indices that will be selected
    unsigned int n, r;
    bool finished; // True if last combination has been reached
};


// Misc functions
double add_logs(double loga, double logb);
double subtract_logs(double loga, double logb);


}

#endif // __BRIDGING_SAMPLER_H__
