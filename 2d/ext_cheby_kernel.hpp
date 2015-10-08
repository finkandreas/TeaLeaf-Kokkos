#ifndef __CHEBYSOLVER
#define __CHEBYSOLVER

#include "ext_chunk.hpp"

/*
 *		CHEBYSHEV SOLVER KERNEL
 */

// Initialises the Cheby solver
template <class Device>
struct ChebyInit
{
    typedef Device device_type;
    typedef Kokkos::View<double*,Device> KView;

    ChebyInit(TLDims dims, KView p, KView r, KView u, KView mi, KView u0, 
            KView w, KView kx, KView ky, double theta, bool preconditioner) 
        : dims(dims), p(p), r(r), u(u), mi(mi), u0(u0), w(w), kx(kx), 
        ky(ky), theta(theta), preconditioner(preconditioner){}

    KOKKOS_INLINE_FUNCTION
        void operator()(const int index) const 
        {
            KOKKOS_INDICES;

            if(INDEX_IN_INNER_DOMAIN)
            {
                const double smvp = SMVP(u);
                w[index] = smvp;
                r[index] = u0[index]-w[index];
                p[index] = (preconditioner ? mi[index]*r[index] : r[index])/theta;
            }
        }

    TLDims dims;
    KView p; 
    KView r; 
    KView u; 
    KView mi; 
    KView u0; 
    KView w; 
    KView kx; 
    KView ky; 
    double theta; 
    bool preconditioner;
};

// Calculates U
template <class Device>
struct ChebyCalcU
{
    typedef Device device_type;
    typedef Kokkos::View<double*,Device> KView;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

    ChebyCalcU(TLDims dims, KView p, KView u) 
        : dims(dims), p(p), u(u) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team) const
    {
        const int team_offset = (team.league_rank() + 2) * dims.y;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2, dims.y-2), [&] (const int &j) {
            const int index = team_offset + j;
            u[index] += p[index];
        });
    }

    TLDims dims;
    KView p; 
    KView u; 
};

// The main Cheby iteration step
template <class Device>
struct ChebyIterate
{
    typedef Device device_type;
    typedef Kokkos::View<double*,Device> KView;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

    ChebyIterate(TLDims dims, KView p, KView r, KView u, KView mi, KView u0, 
            KView w, KView kx, KView ky, KView alphas, KView betas, 
            double step, bool preconditioner) 
        : dims(dims), p(p), r(r), u(u), mi(mi), u0(u0), w(w), kx(kx), ky(ky), 
        alphas(alphas), betas(betas), step(step), preconditioner(preconditioner){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team) const
    {
        const int team_offset = (team.league_rank() + 2) * dims.y;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2, dims.y-2), [&] (const int &j) {
            const int index = team_offset + j;
            const double smvp = SMVP(u);
            w[index] = smvp;
            r[index] = u0[index]-w[index];
            p[index] = alphas[step]*p[index] + betas[step] *
            (preconditioner ? mi[index]*r[index] : r[index]);
        });
    }

    TLDims dims;
    KView p; 
    KView r; 
    KView u; 
    KView mi; 
    KView u0; 
    KView w; 
    KView kx; 
    KView ky; 
    KView alphas; 
    KView betas; 
    int step;
    bool preconditioner;
};

#endif
