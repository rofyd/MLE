COMPILER = nvcc
OBJECTS =  integrator.o gw_output.o utils.o ffteasy.o latticeeasy.o evolution.o initialize.o output.o evolution_GW.o
CFLAGS = -Xcompiler -fopenmp -std=c++17 -O3
LIBS = -lcufft -lgomp
ARCHITECTURE = sm_61
CUDAFLAGS = -arch=$(ARCHITECTURE)
CUDADEBUGFLAGS = -g #-Xptxas -v
DEBUGFLAGS = -g

OBJDIR = ./obj
#_dummy := $(shell mkdir -p $(OBJDIR))

DEPDIR := $(OBJDIR)/.deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d

OBJPROG = $(addprefix $(OBJDIR)/, $(OBJECTS))
VPATH = $(shell find src -type d) variables


all: $(OBJPROG)
	@echo "linking"
	@$(COMPILER) $(LIBS) $(OBJPROG) --gpu-architecture=$(ARCHITECTURE) -o latticeeasy

$(OBJDIR)/%.o: %.cu
$(OBJDIR)/%.o: %.cu $(DEPDIR)/%.d | $(OBJDIR) $(DEPDIR)
	@echo "building" $@
	@$(COMPILER) $(DEPFLAGS) $(CFLAGS) $(CUDADEBUGFLAGS) $(CUDAFLAGS) -I. -c $< -o $@

$(OBJDIR)/%.o: %.cpp
$(OBJDIR)/%.o: %.cpp $(DEPDIR)/%.d | $(OBJDIR) $(DEPDIR) 
	@echo "building" $@
	@$(COMPILER) $(DEPFLAGS) $(DEBUGFLAGS) $(CFLAGS) -I. -x cu -dc $< -o $@

$(OBJDIR):
	mkdir -p $@

$(DEPDIR):
	mkdir -p $@

SRCS := $(shell find \( -name "*.cpp" -o -name "*.cu" \) -exec basename {} .po \;)
DEPFILES := $(SRCS:%.cpp=$(DEPDIR)/%.d) $(SRCS:%.cu=$(DEPDIR)/%.d)
$(DEPFILES):

clean:
	rm -f *.o $(OBJDIR)/*.o app *.dat $(DEPDIR)/*


include $(wildcard $(DEPFILES))
