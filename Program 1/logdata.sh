make clean
make
 
echo
echo --------------------------------------
echo nrows: 1000, ncols: 1000, ncols2: 1000 
echo --------------------------------------

./matmult 1000 1000 1000

for i in 1 2 4 8 12 14 16 20 24 28
do
	export OMP_PROC_BIND=true; export OMP_PLACES=sockets; ./matmult_omp 1000 1000 1000 $i 			
done

echo --------------------------------------
echo nrows: 1000, ncols: 2000, ncols2: 5000 
echo --------------------------------------

./matmult 1000 2000 5000

for i in 1 2 4 8 12 14 16 20 24 28
do
	export OMP_PROC_BIND=true; export OMP_PLACES=sockets; ./matmult_omp 1000 2000 5000 $i			
done

echo --------------------------------------
echo nrows: 9000, ncols: 2500, ncols2: 3750 
echo --------------------------------------

./matmult 9000 2500 3750

for i in 1 2 4 8 12 14 16 20 24 28
do
	export OMP_PROC_BIND=true; export OMP_PLACES=sockets; ./matmult_omp 9000 2500 3750 $i			
done
