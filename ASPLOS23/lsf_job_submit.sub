#!/bin/bash

#BSUB -n 16
#BSUB -q "{gpu_queue_name}"
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -R "span[ptile=8] select[gpu_type=a100]"
#BSUB -o {experiment_name}-%J.out
#BSUB -e {experiment}-%J.err

export PATH=/apps/mpi/gcc/RHEL8/ucx-1.11.0/bin:/apps/mpi/gcc/RHEL8/openmpi-4.1.0-cu111-ucx1110/bin:$PATH
export LD_LIBRARY_PATH=/apps/mpi/gcc/RHEL8/ucx-1.11.0/lib:/apps/mpi/gcc/RHEL8/openmpi-4.1.0-cu111-ucx1110/lib:/usr/lib64/libibverbs:/usr/lib64:/usr/lib:/usr/local:$LD_LIBRARY_PATH

DEBUG="-report-bindings -mca pml_ucx_verbose 100"
MCA_ARG="-mca pml ucx -x UCX_NET_DEVICES=mlx5_0:1"
UCX_OPT="-x NCCL_IB_HCA=mlx5_0:1"

export SINGULARITY_BINDPATH=/etc/libibverbs.d,/usr/lib64/libibverbs,/usr/include/infiniband,/usr/include/rdma,/apps,/LSF,{checkpoint_and_dataset_base_path}
export SINGULARITY_BINDPATH=$SINGULARITY_BINDPATH,$(echo /usr/bin/ib*_* | sed -e 's/ /,/g')
export SINGULARITY_CONTAINLIBS=$(echo /usr/lib64/libmlx*.so* /usr/lib64/librdmacm* /lib64/libib*.so* /usr/lib64/libnl*so* /usr/lib64/libibverbs/libmlx5-rdmav34.so | xargs | sed -e 's/ /,/g')

export GPUS_PER_NODE=8
export HOSTLIST=`echo $LSB_MCPU_HOSTS | sed s/' 8'/' '/g`
export NNODES=`echo $HOSTLIST | wc -w`
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export SINGULARITY_HOST=""

echo "LSB_MCPU_HOSTS: $LSB_MCPU_HOSTS"
echo "LSB_HOSTS: $LSB_HOSTS"
echo "HOSTLIST: $HOSTLIST"

NODE_RANK=0
export node_rankfile=node_rankfile.${LSB_JOBID}
for HOST in $HOSTLIST
do
   if (( $NODE_RANK == 0 )); then
    export MASTER_ADDR=$HOST
    SINGULARITY_HOST="$HOST:1"
    echo "$HOST $NODE_RANK" >> $node_rankfile
   else 
    SINGULARITY_HOST="$SINGULARITY_HOST,$HOST:1"
    echo "$HOST $NODE_RANK" >> $node_rankfile
   fi
   NODE_RANK=$(($NODE_RANK+1))
done

echo "SINGULARITY_HOST: $SINGULARITY_HOST"
echo "NODE_RANK: $NODE_RANK"

mpirun -np $NODE_RANK $DEBUG $MCA_ARG $UCX_OPT -H $SINGULARITY_HOST singularity run --nv --writable-tmpfs --env MASTER_ADDR=$MASTER_ADDR --env node_rankfile=$node_rankfile {singularity_image_path} {sh_script_for_experiment_to_execute} 2>&1 | tee sc22_{experiment_name}_$LSB_JOBID.log

