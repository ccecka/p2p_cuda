For details on this project and its development, visit
https://sites.google.com/site/e205brianesmailp2p/

To compile:
`cd p2p_cuda/unit_tests && make`

To run:
`./test_p2p`

To get timing data:
`nvprof ./test_p2p`


For more profiling and analysis options consider creating a project in Nvidia's Nsight Eclipse Edition and using the built-in visual profiler. 
To do so, open Nsight and create the project (File -> New -> Project... -> Makefile Project with Existing Code) using `p2p_cuda/unit_tests` as the project's root. Now you should be able to simply right click on the project and profile it (Profile As -> Local C/C++ Application).
