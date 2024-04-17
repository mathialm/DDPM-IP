from mpi4py import MPI

if __name__ == "__main__":

    print(f"{MPI.COMM_WORLD.Get_rank()}")