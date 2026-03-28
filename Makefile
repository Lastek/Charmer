CC = cl
CFLAGS = /EHsc /W3 /nologo
LIBS = ole32.lib
TARGET = enumerate_devices.exe
SRC = enumerate_devices.cpp enum_devs2.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) /Fe:$(TARGET) $(SRC) /link $(LIBS)

clean:
	del /Q $(TARGET) *.obj 2>nul

.PHONY: all clean
