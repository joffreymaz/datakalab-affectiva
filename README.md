# To build affectiva-json, first you need to generate the makefiel using CMAKE :
mkdir build && cd build && cmake -DOpenCV_DIR=/usr/ -DBOOST_ROOT=/usr/ -DAFFDEX_DIR=$HOME/affdex-sdk $HOME/sdk-samples ../
# Then, you need to run the makefile to compile affectiva-json.cpp :
cd affectiva-json && make
# To run the program, use :
affectiva-json
