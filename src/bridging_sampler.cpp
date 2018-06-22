
#include "bridging_sampler.h"

#define LOGVERBOSE 0


bridgesamp::BridgingSampler::BridgingSampler(
        uint16_t n_dim,
        uint16_t n_samples,
        std::function<double(const std::vector<uint16_t>&)> logp_node)
    : n_dim(n_dim), n_samples(n_samples),
      eval_node(logp_node),
      r_dim(0,n_dim-1), r_samp(0,n_samples-1),
      r_uniform(0., 1.)
{
    // Seed pseudo-random number generator
    std::random_device rd;
    r.seed(rd());
    
    // Allocate memory to workspaces
    _rand_state_dim_ws.resize(n_dim);
    _gibbs_idx_samp_ws.resize(n_samples);
    _gibbs_lnp_samp_ws.resize(n_samples);
    _gibbs_state_dim_ws.resize(n_dim);
    _transition_state_dim_ws.resize(n_dim);
    _percolate_dim_ws.resize(n_dim);

    // Transition probabilities
    b_prob.reserve(n_dim);
    f_prob.reserve(n_dim);

    b_prob.push_back(0.5);
    f_prob.push_back(0.);
    for(int i=0; i<n_dim-1; i++) {
        b_prob.push_back(0.4);
        f_prob.push_back(0.6);
    }
    b_prob.push_back(0.);
    f_prob.push_back(1.);
    
    // Miscellaneous quantities of use
    p0 = 1.;
    logp0 = 0.;
    log_n_samples = std::log((double)n_samples);
    state_rank = 0;
}


void bridgesamp::BridgingSampler::step() {
    #if LOGVERBOSE
    std::cout << "Entering step()" << std::endl;
    #endif

    // Choose a step type
    double x = r_uniform(r);
    double b = b_prob.at(state_rank);
    double f = f_prob.at(state_rank);
    
    #if LOGVERBOSE
    std::cout << "State rank: " << state_rank << std::endl;
    std::cout << " - b: " << b << std::endl
              << " - f: " << f << std::endl;
    #endif
    

    if(x < b) {
        #if LOGVERBOSE
        std::cout << "Chose to transition backward" << std::endl;
        #endif
        transition_backward();
    } else if(x < f+b) {
        #if LOGVERBOSE
        std::cout << "Chose to transition forward" << std::endl;
        #endif
        transition_forward();
    } else {
        #if LOGVERBOSE
        std::cout << "Chose to take a Gibbs step" << std::endl;
        #endif
        lazy_gibbs_choose_dim();
    }
    
    #if LOGVERBOSE
    std::cout << "Exiting step()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::set_conditional_prob(
    std::function<double(
        uint16_t, // dimension that nodes differ in
        const std::vector<uint16_t>&, // sample # in each dimension
        std::vector<double>&  // holds log(p) of each state
    )> f
) {
    eval_conditional = f;
}


void bridgesamp::BridgingSampler::set_logp0(double _logp0) {
    logp0 = _logp0;
}


uint16_t bridgesamp::BridgingSampler::get_n_dim() const {
    return n_dim;
}


uint16_t bridgesamp::BridgingSampler::get_n_samples() const {
    return n_samples;
}


const std::vector<uint16_t>& bridgesamp::BridgingSampler::get_state() const {
    return state->first;
}


double bridgesamp::BridgingSampler::get_logp() const {
    return state->second.logp;
}


uint16_t bridgesamp::BridgingSampler::get_state_rank() const {
    return state_rank;
}


void bridgesamp::BridgingSampler::randomize_state() {
    #if LOGVERBOSE
    std::cout << "Entering randomize_state()" << std::endl;
    #endif

    // Choose random sample for each dimension
    for(auto& s : _rand_state_dim_ws) {
        s = r_samp(r);
    }
    
    // Find node
    auto it = node.find(_rand_state_dim_ws);

    // Create node if it doesn't exist
    if(it == node.end()) {
        #if LOGVERBOSE
        std::cout << "Creating node";
        for(auto s : _rand_state_dim_ws) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
        #endif

        double logp = eval_node(_rand_state_dim_ws);
        it = node.insert(
            std::make_pair(
                _rand_state_dim_ws,
                bridgesamp::Node{std::exp(logp), logp}
            )
        ).first;

        percolate_up(it);
    }

    // Update the state to point to this node
    state = it;
    state_rank = 0; // At 0th level in hierarchy
    
    #if LOGVERBOSE
    std::cout << "Exiting randomize_state()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::_lazy_gibbs_inner(
        std::vector<uint16_t>& starting_state,
        uint16_t dim)
{
    #if LOGVERBOSE
    std::cout << "Entering _lazy_gibbs_inner()" << std::endl;
    #endif

    #if LOGVERBOSE
    std::cout << "Starting state:";
    for(auto s : starting_state) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif

    // Find ln(p) of states that have already been explored
    _gibbs_idx_samp_ws.clear(); // Will hold explored sample numbers
    _gibbs_lnp_samp_ws.clear(); // Will hold corresponding ln(p)

    for(uint16_t s=0; s<n_samples; s++) {
        starting_state[dim] = s;

        auto it = node.find(starting_state);
        if(it != node.end()) {
            _gibbs_idx_samp_ws.push_back(s);
            _gibbs_lnp_samp_ws.push_back(it->second.logp);
        }
    }

    uint16_t n_unexplored = n_samples - _gibbs_idx_samp_ws.size();

    #if LOGVERBOSE
    std::cout << n_unexplored << " of " << n_samples << " states unexplored." << std::endl;
    #endif
    
    // Get maximum ln(p) of explored states
    double lnp_max = *(std::max_element(
        _gibbs_lnp_samp_ws.begin(),
        _gibbs_lnp_samp_ws.end())
    );
    
    // Find total ln(p) of explored states
    double lnp_explored = 0.;

    for(auto lnp : _gibbs_lnp_samp_ws) {
        lnp_explored += std::exp(lnp - lnp_max);
    }

    lnp_explored = std::log(lnp_explored) + lnp_max;

    // Calculate total ln(p) of unexplored states
    double ln_n_unexplored = std::log((double)(n_samples-_gibbs_idx_samp_ws.size()));
    double lnp_unexplored = get_lnp0_of_rank(state_rank) + ln_n_unexplored;

    double lnp_tot = add_logs(lnp_unexplored, lnp_explored);

    #if LOGVERBOSE
    std::cout << "p(explored) = "
              << std::exp(lnp_explored-lnp_tot) << std::endl;
    #endif

    // Decide whether to pick explored state
    if(!n_unexplored || (std::log(r_uniform(r)) < lnp_explored-lnp_tot)) {
        #if LOGVERBOSE
        std::cout << "Picking explored state." << std::endl;
        #endif

        // Pick explored state
        for(auto& lnp : _gibbs_lnp_samp_ws) {
            lnp = std::exp(lnp - lnp_max);
        }
        std::discrete_distribution<uint16_t> d(
            _gibbs_lnp_samp_ws.begin(),
            _gibbs_lnp_samp_ws.end()
        );

        starting_state[dim] = _gibbs_idx_samp_ws[d(r)];
        state = node.find(starting_state);
        
        #if LOGVERBOSE
        std::cout << "Picked explored state:";
        for(auto s : starting_state) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
        #endif
    } else {
        #if LOGVERBOSE
        std::cout << "Picking unexplored state." << std::endl;
        #endif

        // Pick unexplored state at random
        std::uniform_int_distribution<uint16_t> d(0, n_unexplored-1);
        uint16_t i_pick = d(r);

        #if LOGVERBOSE
        std::cout << "Scanning for unexplored state #"
                  << i_pick << ":" << std::endl;
        #endif

        // Scan through to find i_pick'th unexplored sample
        int32_t idx = 0;
        int32_t i_unexplored = -1;
        auto it_explored = _gibbs_idx_samp_ws.begin();
        for(; i_unexplored != i_pick; idx++) {
            #if LOGVERBOSE
            std::cout << "- sample " << idx;
            #endif
            if((it_explored != _gibbs_idx_samp_ws.end()) && (*it_explored == idx)) {
                #if LOGVERBOSE
                std::cout << ": explored" << std::endl;
                #endif
                ++it_explored;
            } else {
                #if LOGVERBOSE
                std::cout << ": unexplored" << std::endl;
                #endif
                ++i_unexplored;
            }
        }

        // Add the new state
        starting_state[dim] = idx-1;
        state = get_node(starting_state);
        
        #if LOGVERBOSE
        std::cout << "Picked unexplored state:";
        for(auto s : starting_state) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
        #endif
    }

    #if LOGVERBOSE
    std::cout << "Exiting _lazy_gibbs_inner()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::gibbs(uint16_t dim) {
    #if LOGVERBOSE
    std::cout << "Entering gibbs()" << std::endl;
    #endif
    
    if(state_rank != 0) {
        #if LOGVERBOSE
        std::cout << "Can only take Gibbs step at zeroeth level in hierarchy."
                  << std::endl;
        std::cout << "Exiting gibbs()" << std::endl;
        #endif
        return;
    }
    
    // Copy current state into workspace
    _gibbs_state_dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_gibbs_state_dim_ws)
    );

    #if LOGVERBOSE
    std::cout << "Starting state:";
    for(auto s : _gibbs_state_dim_ws) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif

    // Find out which nodes have been explored, and which not
    _gibbs_idx_samp_ws.clear(); // Will hold samples corresponding to unexplored nodes

    NodeMap::iterator it;

    if(eval_conditional) {
        #if LOGVERBOSE
        std::cout << "Using conditional probability function." << std::endl;
        #endif

        for(int k=0; k<n_samples; k++) {
            _gibbs_state_dim_ws[dim] = k;

            // Try to find node
            it = node.find(_gibbs_state_dim_ws);

            if(it == node.end()) { // Node not found
                _gibbs_idx_samp_ws.push_back(k);
                _gibbs_lnp_samp_ws.push_back(0.); // dummy value, update later
            } else {
                _gibbs_lnp_samp_ws.push_back(it->second.logp);
            }
        }

        #if LOGVERBOSE
        std::cout << _gibbs_idx_samp_ws.size() << " of " << n_samples
                  << "states unexplored." << std::endl;
        #endif

        // Determine log(p) of unexplored states using conditional prob. function
        _gibbs_lnp_samp_ws.resize(n_samples); // Will hold log(p) of each node
        if(_gibbs_idx_samp_ws.size()) {
            eval_conditional(dim, _gibbs_state_dim_ws, _gibbs_lnp_samp_ws);
        }

        // Add nodes, and percolate log(p) upward
        double logp;
        for(auto s : _gibbs_idx_samp_ws) {
            _gibbs_state_dim_ws[dim] = s;
            logp = _gibbs_lnp_samp_ws[s];

            it = node.insert(
                std::make_pair(
                    _gibbs_state_dim_ws,
                    bridgesamp::Node{std::exp(logp), logp}
                )
            ).first;

            // Propagate change in log(p) upward
            percolate_up(it);
        }
    } else {
        #if LOGVERBOSE
        std::cout << "No conditional probability function set." << std::endl;
        #endif

        _gibbs_lnp_samp_ws.clear(); 

        for(int k=0; k<n_samples; k++) {
            _gibbs_state_dim_ws[dim] = k;

            // Get or create node
            it = get_node(_gibbs_state_dim_ws);
            _gibbs_lnp_samp_ws.push_back(it->second.logp);
        }
    }

    // Get maximum log(p), and transform log(p) to p/p_max.
    double logp_max = *std::max(_gibbs_lnp_samp_ws.begin(), _gibbs_lnp_samp_ws.end());
    for(auto& p : _gibbs_lnp_samp_ws) {
        p = std::exp(p - logp_max);
    }
    
    #if LOGVERBOSE
    std::cout << "log(p_max) = " << logp_max << std::endl;
    #endif
    
    std::discrete_distribution<uint16_t> d(
        _gibbs_lnp_samp_ws.begin(),
        _gibbs_lnp_samp_ws.end());
    _gibbs_state_dim_ws[dim] = d(r);

    #if LOGVERBOSE
    std::cout << "Chose sample #" << _gibbs_state_dim_ws[dim] << std::endl;
    #endif

    state = get_node(_gibbs_state_dim_ws);

    // TODO: Bulk percolate-up function?

    #if LOGVERBOSE
    std::cout << "Exiting gibbs()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::lazy_gibbs(uint16_t dim) {
    #if LOGVERBOSE
    std::cout << "Entering lazy_gibbs()" << std::endl;
    #endif

    if(state->first[dim] == n_samples) {
        #if LOGVERBOSE
        std::cout << "Cannot take Gibbs step in empty dimension." << std::endl;
        std::cout << "Exiting lazy_gibbs()" << std::endl;
        #endif
        return;
    }

    // Copy current state into workspace
    _gibbs_state_dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_gibbs_state_dim_ws)
    );

    _lazy_gibbs_inner(_gibbs_state_dim_ws, dim);

    #if LOGVERBOSE
    std::cout << "Exiting lazy_gibbs()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::lazy_gibbs_choose_dim() {
    #if LOGVERBOSE
    std::cout << "Entering lazy_gibbs_choose_dim()" << std::endl;
    #endif

    if(state_rank == n_dim) {
        #if LOGVERBOSE
        std::cout << "Cannot take a Gibbs step at the top level of the hierarchy."
                  << std::endl;
        std::cout << "Exiting lazy_gibbs_choose_dim()" << std::endl;
        #endif
        return;
    }

    // Choose which dimension to step in
    std::uniform_int_distribution<uint16_t> d(0, n_dim-state_rank-1);
    uint16_t i_pick = d(r);
    
    #if LOGVERBOSE
    std::cout << "Chose non-empty dimension #" << i_pick << std::endl;
    #endif

    int32_t i_nonempty = -1;
    uint16_t dim = 0;
    for(auto it=state->first.begin(); i_nonempty != i_pick; ++it, dim++) {
        if(*it != n_samples) {
            i_nonempty++;
        }
    }

    #if LOGVERBOSE
    std::cout << " -> dim = " << dim << std::endl;
    #endif

    if(state_rank == 0) {
        gibbs(dim);
    } else {
        lazy_gibbs(dim);
    }

    #if LOGVERBOSE
    std::cout << "Exiting lazy_gibbs_choose_dim()" << std::endl;
    #endif
}


bridgesamp::NodeMap::const_iterator bridgesamp::BridgingSampler::cbegin() const {
    return node.cbegin();
}


bridgesamp::NodeMap::const_iterator bridgesamp::BridgingSampler::cend() const {
    return node.cend();
}


uint16_t bridgesamp::BridgingSampler::get_n_empty(
        const std::vector<uint16_t>& samp)
{
    uint16_t n_empty = 0;
    for(auto s : samp) {
        if(s == n_samples) {
            n_empty++;
        }
    }
    return n_empty;
}


double bridgesamp::BridgingSampler::get_lnp0_of_rank(uint16_t rank) {
    return logp0 + rank * log_n_samples;
}


bridgesamp::NodeMap::iterator bridgesamp::BridgingSampler::get_node(
        const std::vector<uint16_t>& samp)
{
    #if LOGVERBOSE
    std::cout << "Entering get_node()" << std::endl;
    #endif

    // Find node
    auto it = node.find(samp);

    // Get rank of node
    uint16_t n_empty = get_n_empty(samp);
    #if LOGVERBOSE
    std::cout << "# empty: " << n_empty << std::endl;
    #endif

    // Create node if it doesn't exist
    if(it == node.end()) {
        #if LOGVERBOSE
        std::cout << "Creating node";
        for(auto s : samp) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
        #endif

        // Calculate probability of new node
        double logp;
        if(n_empty == 0) {
            // Evaluate rank-0 nodes exactly
            logp = eval_node(samp);
            #if LOGVERBOSE
            std::cout << "Evaluated log(p) exactly: " << logp << std::endl;
            #endif
            
            // Insert the node, and get iterator to it
            it = node.insert(
                std::make_pair(samp, bridgesamp::Node{std::exp(logp), logp})
            ).first;

            // Propagate change in log(p) upward
            percolate_up(it);
        } else {
            // For higher-rank nodes, assume constant probability for
            // all rank-0 child nodes. Probability = const * (# children),
            // where # of children = (# of samples)^rank.
            //logp = logp0 + n_empty * log_n_samples;
            logp = get_lnp0_of_rank(n_empty);
            #if LOGVERBOSE
            std::cout << "Set log(p) to default: " << logp << std::endl;
            #endif
            
            // Insert the node, and get iterator to it
            it = node.insert(
                std::make_pair(samp, bridgesamp::Node{std::exp(logp), logp})
            ).first;
        }
    }

    #if LOGVERBOSE
    std::cout << "Exiting get_node()" << std::endl;
    #endif

    return it;
}


double bridgesamp::add_logs(double loga, double logb) {
    // Numerically stable way to add <a> and <b>, where their logarithms
    // are taken as input, and log(a+b) is given as output.
    if(loga > logb) {
        return loga + std::log(1. + std::exp(logb-loga));
    } else {
        return logb + std::log(1. + std::exp(loga-logb));
    }
}


double bridgesamp::subtract_logs(double loga, double logb) {
    // Numerically stable way to subtract <b> from <a>, where their logarithms
    // are taken as input, and log(a-b) is given as output. If b>a, then
    // returns -infinity.
    if(loga > logb) {
        return loga + std::log(1. - std::exp(logb-loga));
    } else {
        return -std::numeric_limits<double>::infinity();
    }
}


void bridgesamp::BridgingSampler::transition_backward() {
    #if LOGVERBOSE
    std::cout << "Entering transition_backward()" << std::endl;
    #endif
    
    #if LOGVERBOSE
    std::cout << "Starting state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif

    if(state_rank == n_dim) {
        #if LOGVERBOSE
        std::cout << "Already at highest level in hierarchy." << std::endl
                  << "Exiting transition_backward()" << std::endl;
        #endif
        return;
    }
    
    // Find non-empty dimensions
    get_nonempty_dims(state, _transition_state_dim_ws);
    uint16_t n_nonempty = _transition_state_dim_ws.size();
    #if LOGVERBOSE
    std::cout << n_nonempty << " non-empty dimensions" << std::endl;
    #endif

    // Choose a dimension to blank
    std::uniform_int_distribution<uint16_t> d(0, n_nonempty-1);
    uint16_t idx = _transition_state_dim_ws[d(r)];
    
    #if LOGVERBOSE
    std::cout << "Blanking dimension " << idx << std::endl;
    #endif

    // Copy current state into workspace
    _transition_state_dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_transition_state_dim_ws)
    );
    
    // Blank the chosen dimension and transition
    _transition_state_dim_ws[idx] = n_samples;
    state = get_node(_transition_state_dim_ws);
    state_rank++;
    
    #if LOGVERBOSE
    std::cout << "New state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif
    
    #if LOGVERBOSE
    std::cout << "Exiting transition_backward()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::transition_forward() {
    #if LOGVERBOSE
    std::cout << "Entering transition_forward()" << std::endl;
    #endif
    
    #if LOGVERBOSE
    std::cout << "Starting state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif

    if(state_rank == 0) {
        #if LOGVERBOSE
        std::cout << "Already at lowest level in hierarchy." << std::endl
                  << "Exiting transition_forward()" << std::endl;
        #endif
        return;
    }
    
    // Find empty dimensions
    get_empty_dims(state, _transition_state_dim_ws);
    uint16_t n_empty = _transition_state_dim_ws.size();
    #if LOGVERBOSE
    std::cout << n_empty << " empty dimensions" << std::endl;
    #endif

    // Choose a dimension to fill
    std::uniform_int_distribution<uint16_t> d(0, n_empty-1);
    uint16_t idx = _transition_state_dim_ws[d(r)];
    
    #if LOGVERBOSE
    std::cout << "Filling dimension " << idx << std::endl;
    #endif

    // Copy current state into workspace
    _transition_state_dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_transition_state_dim_ws)
    );
    
    // Fill the chosen dimension with zero
    _transition_state_dim_ws[idx] = 0;
    state_rank--;

    // Take a Gibbs step
    _lazy_gibbs_inner(_transition_state_dim_ws, idx);
    
    #if LOGVERBOSE
    std::cout << "New state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif
    
    #if LOGVERBOSE
    std::cout << "Exiting transition_forward()" << std::endl;
    #endif
}


void bridgesamp::BridgingSampler::percolate_up(const NodeMap::iterator& n) {
    #if LOGVERBOSE
    std::cout << "Entering percolate_up()" << std::endl;
    #endif
    
    #if LOGVERBOSE
    std::cout << "Starting from state:";
    for(auto s : n->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    #endif

    // Find non-empty dimensions
    get_nonempty_dims(n, _percolate_dim_ws);
    #if LOGVERBOSE
    std::cout << _percolate_dim_ws.size()
              << " non-empty dimensions" << std::endl;
    #endif

    // Determine change in probability to propagate
    double dlogp; // log(|dp|)
    bool positive; // True if change is positive
    if(n->second.logp >= logp0) {
        dlogp = subtract_logs(n->second.logp, logp0);
        positive = true;
    } else {
        dlogp = subtract_logs(logp0, n->second.logp);
        positive = false;
    }

    #if LOGVERBOSE
    std::cout << "Change in probability:" << std::endl
              << "        dp = " << (positive ? "+" : "-")
                                 << std::exp(dlogp) << std::endl
              << " log(|dp|) = " << dlogp << " ("
                                 << (positive ? "+" : "-")
                                 << ")" << std::endl;
    #endif

    // Loop through higher ranks in hierarchy
    std::vector<unsigned int> idx;
    std::vector<uint16_t> node_key;
    idx.reserve(n_dim);
    node_key.reserve(n_dim);

    for(int rank=1; rank<=_percolate_dim_ws.size(); rank++) {
        // Blank out every combination of <rank> entries, obtaining
        // keys to all parent nodes
        CombinationGenerator cgen(_percolate_dim_ws.size(), rank);
        while(cgen.next(idx)) {
            #if LOGVERBOSE
            std::cout << "Blanking dimensions:";
            for(auto i : idx) {
                std::cout << " " << i;
            }
            std::cout << std::endl;
            #endif

            // Copy original key into node_key
            node_key.clear();
            std::copy(
                n->first.begin(),
                n->first.end(),
                std::back_inserter(node_key)
            );
            // Blank selected dimensions
            for(auto i : idx) {
                node_key[i] = n_samples;
            }
            #if LOGVERBOSE
            std::cout << "Percolate to =";
            for(auto i : node_key) {
                std::cout << " " << i;
            }
            std::cout << std::endl;
            #endif

            auto it = get_node(node_key); // Iterator to parent node

            // Update probability of parent node
            if(positive) {
                it->second.logp = add_logs(it->second.logp, dlogp);
            } else {
                it->second.logp = subtract_logs(it->second.logp, dlogp);
            }
        }
    }
    
    #if LOGVERBOSE
    std::cout << "Exiting percolate_up()" << std::endl;
    #endif
}


double bridgesamp::BridgingSampler::n_children(uint16_t order) {
    return pow((double)(n_samples+1), (double)order);
}


void bridgesamp::BridgingSampler::get_nonempty_dims(
        const NodeMap::iterator& n,
        std::vector<uint16_t>& dims_out)
{
    dims_out.clear();
    dims_out.reserve(n_dim);

    for(int i=0; i<n_dim; i++) {
        if(n->first[i] != n_samples) {
            dims_out.push_back(i);
        }
    }
}


void bridgesamp::BridgingSampler::get_empty_dims(
        const NodeMap::iterator& n,
        std::vector<uint16_t>& dims_out)
{
    dims_out.clear();
    dims_out.reserve(n_dim);

    for(int i=0; i<n_dim; i++) {
        if(n->first[i] == n_samples) {
            dims_out.push_back(i);
        }
    }
}


double bridgesamp::BridgingSampler::fill_factor() const {
    uint64_t n_obj = node.size();
    double n_obj_max = std::pow((double)n_samples+1., (double)n_dim);

    return (double)n_obj / n_obj_max;
}



/*
 *  Combination generator - cycles through combinations (n choose r)
 */

bridgesamp::CombinationGenerator::CombinationGenerator(
        unsigned int n, unsigned int r)
    : r(r), n(n), mask(n, false), finished(false)
{
    assert(n >= r);
    std::fill(mask.end()-r, mask.end(), true);
}


void bridgesamp::CombinationGenerator::reset() {
    std::fill(mask.begin(), mask.end()-r-1, false);
    std::fill(mask.end()-r, mask.end(), true);
    finished = false;
}


bool bridgesamp::CombinationGenerator::next(std::vector<unsigned int>& out_idx) {
    if(finished) { return false; }
    
    out_idx.resize(r);
    //assert(out_idx.size() >= r);

    unsigned int k = 0;
    for(unsigned int i=0; i<n; i++) {
        if(mask[i]) {
            out_idx[k] = i;
            k++;
        }
    }

    finished = !std::next_permutation(mask.begin(), mask.end());

    return true;
}


std::size_t bridgesamp::VectorHasher::operator()(
        const std::vector<uint16_t>& vec) const
{
    std::size_t seed = vec.size();
    for(auto& i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

