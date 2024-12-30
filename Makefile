CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O3
DEBUGFLAGS = -g -DDEBUG

TARGET = power_method
SRCS = power_method.cpp
OBJS = $(SRCS:.cpp=.o)

ifeq ($(OS),Windows_NT)
    RM = del /Q /F
    EXE_EXT = .exe
else
    RM = rm -rf
    EXE_EXT =
endif

TARGET_BIN = $(TARGET)$(EXE_EXT)

.PHONY: all clean debug

all: $(TARGET_BIN)

$(TARGET_BIN): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET_BIN) $(CXXFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

debug: CXXFLAGS += $(DEBUGFLAGS)
debug: clean $(TARGET_BIN)

clean:
	$(RM) $(TARGET_BIN) *.o
