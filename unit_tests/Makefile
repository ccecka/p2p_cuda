EXEC = test_p2p

include ../Makefile.inc

# Extra dependencies for executables
# TODO: Automatically detect a .kern file in the dependencies?

# List the Kernel dependency (.kern) for each executable

# Test P2P should have all kernels built?
test_p2p:        $(patsubst %.kern,%.o,$(wildcard $(KERNEL_DIR)/*.kern))
