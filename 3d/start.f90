!Crown Copyright 2014 AWE.
!
! This file is part of TeaLeaf.
!
! TeaLeaf is free software: you can redistribute it and/or modify it under 
! the terms of the GNU General Public License as published by the 
! Free Software Foundation, either version 3 of the License, or (at your option) 
! any later version.
!
! TeaLeaf is distributed in the hope that it will be useful, but 
! WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
! FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
! details.
!
! You should have received a copy of the GNU General Public License along with 
! TeaLeaf. If not, see http://www.gnu.org/licenses/.

!>  @brief Main set up routine
!>  @author David Beckingsale, Wayne Gaudin
!>  @details Invokes the mesh decomposer and sets up chunk connectivity. It then
!>  allocates the communication buffers and call the chunk initialisation and
!>  generation routines and primes the halo cells and writes an initial field summary.

SUBROUTINE start

  USE tea_module
  USE parse_module
  USE update_halo_module

  IMPLICIT NONE

  INTEGER :: c
  INTEGER :: x_cells,y_cells,z_cells
  INTEGER, ALLOCATABLE :: right(:),left(:),top(:),bottom(:),back(:),front(:)
  INTEGER :: fields(NUM_FIELDS)
  LOGICAL :: profiler_off

  IF(parallel%boss)THEN
    WrITE(g_out,*) 'Setting up initial geometry'
    WRITE(g_out,*)
  ENDIF

  time  = 0.0
  step  = 0
  dt    = dtinit

  CALL tea_barrier

  CALL tea_get_num_chunks(number_of_chunks)

  ALLOCATE(chunks(1:number_of_chunks))
  ALLOCATE(left(1:number_of_chunks))
  ALLOCATE(right(1:number_of_chunks))
  ALLOCATE(bottom(1:number_of_chunks))
  ALLOCATE(top(1:number_of_chunks))
  ALLOCATE(back(1:number_of_chunks))
  ALLOCATE(front(1:number_of_chunks))

  CALL tea_decompose(grid%x_cells,grid%y_cells, grid%z_cells,left,right,bottom,top, back, front)

  DO c=1,chunks_per_task
      
    ! Needs changing so there can be more than 1 chunk per task
    chunks(c)%task = parallel%task

    x_cells = right(c) -left(c)  +1
    y_cells = top(c)   -bottom(c)+1
    z_cells = front(c) -back(c)  +1

   IF(chunks(c)%task.EQ.parallel%task)THEN
      CALL build_field(c,x_cells,y_cells,z_cells)

      IF(use_ext_kernels) THEN
        CALL ext_init(x_cells, y_cells, z_cells, parallel%task, &
            chunks(c)%field%density, &
            chunks(c)%field%energy0, &
            chunks(c)%field%energy1, &
            chunks(c)%field%vector_Kx, &
            chunks(c)%field%vector_Ky, &
            chunks(c)%field%vector_Kz, &
            chunks(c)%field%vector_Mi, &
            chunks(c)%field%vector_p, &
            chunks(c)%field%vector_r, &
            chunks(c)%field%vector_sd, &
            chunks(c)%field%u, &
            chunks(c)%field%u0, &
            chunks(c)%field%vector_w, &
            chunks(c)%field%vector_z)
      ENDIF
    ENDIF
    
    chunks(c)%field%left    = left(c)
    chunks(c)%field%bottom  = bottom(c)
    chunks(c)%field%right   = right(c)
    chunks(c)%field%top     = top(c)
    chunks(c)%field%back    = back(c)
    chunks(c)%field%front   = front(c)
    chunks(c)%field%left_boundary   = 1
    chunks(c)%field%bottom_boundary = 1
    chunks(c)%field%back_boundary   = 1
    chunks(c)%field%right_boundary  = grid%x_cells
    chunks(c)%field%top_boundary    = grid%y_cells
    chunks(c)%field%front_boundary  = grid%z_cells
    chunks(c)%field%x_min = 1
    chunks(c)%field%y_min = 1
    chunks(c)%field%z_min = 1
    chunks(c)%field%x_max = right(c)-left(c)+1
    chunks(c)%field%y_max = top(c)-bottom(c)+1
    chunks(c)%field%z_max = front(c)-back(c)+1
   
  ENDDO

   
  DEALLOCATE(left,right,bottom,top,back,front)

  CALL tea_barrier

  DO c=1,chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task)THEN
      CALL tea_allocate_buffers(c)
    ENDIF
  ENDDO

  DO c=1,chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task)THEN
      CALL initialise_chunk(c)
    ENDIF
  ENDDO

  IF(parallel%boss)THEN
     WRITE(g_out,*) 'Generating chunks'
  ENDIF

  DO c=1,chunks_per_task
    IF(chunks(c)%task.EQ.parallel%task)THEN
        CALL generate_chunk(c)
    ENDIF
  ENDDO

  CALL tea_barrier

  ! Do not profile the start up costs so totals add up 
  profiler_off=profiler_on
  profiler_on=.FALSE.

  ! Prime all halo data for the first step
  fields=0
  fields(FIELD_DENSITY)=1
  fields(FIELD_ENERGY0)=1
  fields(FIELD_ENERGY1)=1

  CALL update_halo(fields,2)

  IF(parallel%boss)THEN
     WRITE(g_out,*)
     WRITE(g_out,*) 'Problem initialised and generated'
  ENDIF

  CALL field_summary()

  IF(visit_frequency.NE.0) CALL visit()

  CALL tea_barrier

  profiler_on=profiler_off

END SUBROUTINE start
