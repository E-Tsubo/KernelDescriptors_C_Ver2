all:lib test computekdes mattest matop
lib: libkerneldesc.cc
	g++ -g -c libkerneldesc.cc -msse -msse2 -msse3 -msse4  -IInclude -lmatio `pkg-config opencv matio --cflags --libs`
#	g++ -g -c libkerneldesc.cc -msse -msse2 -msse3 -msse4  -IInclude -lmatio `pkg-config opencv matio --cflags --libs`
	ar rvs libkerneldesc.a libkerneldesc.o
test: kernelmain.cc
	g++ -fpermissive -O3 -g -msse -msse2 -msse3 -msse4 -o kernelmain kernelmain.cc libkerneldesc.a liblinear-dense-float/tron.o liblinear-dense-float/blas/blas.a liblinear-dense-float/linear.o -IInclude -lmatio -lboost_thread -lboost_filesystem -lboost_system `pkg-config opencv matio --cflags --libs`
#	g++ -g -o kernelmain kernelmain.cc libkerneldesc.a -IInclude -lmatio -lboost_thread -lboost_filesystem -lboost_system `pkg-config opencv matio --cflags --libs`
#	g++ -O2 -o kernelmain -lmatio -lhdf5 -lcv -lhighgui -IInclude kernelmain.cc
#	g++ -O3 -o kernelmain kernelmain.cc -IInclude -lmatio -fopenmp `pkg-config opencv matio --cflags --libs`
#	g++ -O3 -msse -msse2 -msse3 -msse4 -o kernelmain kernelmain.cc -IInclude -lmatio -fopenmp `pkg-config opencv matio --cflags --libs`
#	g++ -g -o kernelmain kernelmain.cc libkerneldesc.cc -IInclude -lmatio `pkg-config opencv matio --cflags --libs`
#	g++ -O3 -msse -msse2 -msse3 -msse4 -o kernelmain kernelmain.cc libkerneldesc.cc -IInclude -lmatio `pkg-config opencv matio --cflags --libs`
computekdes: computekdes.cc
	g++ -fpermissive -O3 -g -msse -msse2 -msse3 -msse4 -o computekdes computekdes.cc libkerneldesc.a liblinear-dense-float/tron.o liblinear-dense-float/blas/blas.a liblinear-dense-float/linear.o -IInclude -lmatio -lboost_thread -lboost_filesystem -lboost_system `pkg-config opencv matio --cflags --libs`
mattest: mattest.cc
	g++ -O2 -g -o mattest mattest.cc `pkg-config matio --cflags --libs` -lhdf5 -IInclude 
matop: matop.cc
	g++ -O3 -g -o matop -fopenmp -IInclude matop.cc
clean:
	rm kernelmain mattest matop libkerneldesc.o libkerneldesc.a computekdes

