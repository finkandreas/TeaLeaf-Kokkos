#include "ext_chunk.hpp"

/*
 *		PPCG SOLVER KERNEL
 */

// Initialises Sd
template <class Device>
struct PPCGInitSd
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;

	PPCGInitSd(TLDims dims, KView sd, KView r, KView mi, double theta, bool preconditioner) 
		: dims(dims), sd(sd), r(r), mi(mi), theta(theta), preconditioner(preconditioner){}

	KOKKOS_INLINE_FUNCTION
	void operator()(const int index) const 
    {
		KOKKOS_INDICES;

		if(INDEX_IN_INNER_DOMAIN)
		{
			sd[index] = (preconditioner ? mi[index]*r[index] : r[index]) / theta;
		}
	}

	TLDims dims;
	KView sd; 
	KView mi; 
	KView r; 
	double theta; 
	bool preconditioner;
};

// Calculates U and R
template <class Device>
struct PPCGCalcU
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

	PPCGCalcU(TLDims dims, KView sd, KView r, KView u, KView kx, KView ky) 
		: dims(dims), sd(sd), r(r), u(u), kx(kx), ky(ky) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team) const
    {
        const int team_offset = (team.league_rank() + 2) * dims.y;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2, dims.y-2), [&] (const int &j) {
            const int index = team_offset + j;
			const double smvp = SMVP(sd);
			r[index] -= smvp;
			u[index] += sd[index];
        });
    }

	TLDims dims;
	KView sd; 
	KView r; 
	KView u; 
	KView kx; 
	KView ky; 
};

// Calculates Sd
template <class Device>
struct PPCGCalcSd
{
	typedef Device device_type;
	typedef Kokkos::View<double*,Device> KView;
    typedef Kokkos::TeamPolicy<Device> team_policy;
    typedef typename team_policy::member_type team_member;

	PPCGCalcSd(TLDims dims, KView sd, KView r, KView mi, KView alphas, 
			KView betas, double theta, bool preconditioner, int step) 
		: dims(dims), sd(sd), r(r), mi(mi), alphas(alphas), betas(betas),
	   	theta(theta), preconditioner(preconditioner), step(step){}

    KOKKOS_INLINE_FUNCTION
    void operator()(const team_member& team) const
    {
        const int team_offset = (team.league_rank() + 2) * dims.y;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 2, dims.y-2), [&] (const int &j) {
            const int index = team_offset + j;
			sd[index] = alphas[step]*sd[index]+betas[step]*
				(preconditioner ? mi[index]*r[index] : r[index]);
        });
    }

	TLDims dims;
	KView sd; 
	KView mi; 
	KView r; 
	KView alphas; 
	KView betas; 
	double theta; 
	bool preconditioner;
	int step;
};

