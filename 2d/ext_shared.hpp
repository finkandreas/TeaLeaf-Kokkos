#ifndef __SHARED
#define __SHARED

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>
#include "ext_profiler.hpp"

#define HALO_PAD 2
#define HP2 HALO_PAD*2
#define CHUNK_LEFT 1
#define CHUNK_RIGHT 2
#define CHUNK_BOTTOM 3
#define CHUNK_TOP 4
#define NUM_FACES 4
#define EXTERNAL_FACE -1

#define FIELD_DENSITY 1
#define FIELD_ENERGY0 2
#define FIELD_ENERGY1 3
#define FIELD_U 4
#define FIELD_P 5
#define FIELD_SD 6
#define NUM_FIELDS 6
#define MAX_DEPTH 2

#define CONDUCTIVITY 1
#define RECIP_CONDUCTIVITY 2

#ifdef CUDA
	#define DEVICE Kokkos::Cuda
#else
	#define DEVICE Kokkos::OpenMP
#endif

#define SMVP(a) \
	(1.0 + (kx[index+1]+kx[index])\
	 + (ky[index+dims.x]+ky[index]))*a[index]\
	 - (kx[index+1]*a[index+1]+kx[index]*a[index-1])\
	 - (ky[index+dims.x]*a[index+dims.x]+ky[index]*a[index-dims.x]);

#define KOKKOS_INDICES \
	int kk = index % dims.x; \
	int jj = index / dims.x; \

#define INDEX_IN_INNER_DOMAIN \
	kk >= HALO_PAD && kk < dims.x-HALO_PAD && \
	jj >= HALO_PAD && jj < dims.y-HALO_PAD

#define INDEX_IN_ONE_DOMAIN \
	jj > 0 && jj < dims.y-1 && \
	kk > 0 && kk < dims.x-1

#define INDEX_SKEW_DOMAIN \
	jj >= HALO_PAD && jj < dims.y-1 && \
	kk >= HALO_PAD && kk < dims.x-1

struct TLDims
{
	int x;
	int y;
	int z;
};

class KokkosHelper
{
	public:
        // Used to manage copying the raw pointers
		template <class Type>
		static void InitMirror(const Kokkos::View<Type*,Kokkos::HostSpace>& mirror, const Type* buffer, int len)
		{
			for(int kk = 0; kk != len; ++kk)
			{
				mirror(kk) = buffer[kk];
			}
		}
};

#endif

