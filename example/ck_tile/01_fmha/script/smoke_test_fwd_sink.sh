#!/bin/bash
# TODO: run this script from CK root or build directory
#EXE="/code/composable_kernel/build/bin/tile_example_fmha_fwd"
EXE="$(find . -name tile_example_fmha_fwd -type f | head -n 1)"
KNAME=1


COMMON_ARGS='-v=1 -warmup=0 -repeat=1'


#    x=4/y=1                   
#    1 1 1 1 * * * *           1 1 1 1 * * * * 
#    * 1 1 1 1 * * *           1 1 1 1 1 * * *
#    * * 1 1 1 1 * *   ---->   1 1 1 1 1 1 * *
#    * * * 1 1 1 1 *           1 1 * 1 1 1 1 *
#    * * * * 1 1 1 1           1 1 * * 1 1 1 1 
#    l=0/r=3(tl)               l=0/r=3/s=2(tl)
#    l=3/r=0(br)               l=3/r=0/s=2(br)  
                    

$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=5 -s_k=8 -bias=n -lse=0 -iperm=0 -operm=0 -mask=t:0,3,2 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0  -kname=$KNAME $COMMON_ARGS  

$EXE -prec=fp16 -mode=1 -b=1 -h=1 -d=16 -d_v=32 -s=5 -s_k=8 -bias=n -lse=0 -iperm=0 -operm=0 -mask=b:3,0,2 -vlayout=c -num_splits=1 -page_block_size=0 -cache_batch_idx=0 -kname=$KNAME $COMMON_ARGS  

$EXE -prec=fp16 -mode=1 -b=2 -h=1 -d=64 -s=1024 -bias=n -p_drop=0.0 -lse=1 -iperm=0 -operm=0 -vlayout=r -mask=t:-1,4,4 -kname=$KNAME $COMMON_ARGS  

$EXE -prec=fp16 -mode=1 -b=1 -h=2 -d=64 -s=1024 -bias=n -p_drop=0.0 -lse=1 -iperm=0 -operm=0 -vlayout=r -mask=b:-1,7,4 -kname=$KNAME $COMMON_ARGS  
