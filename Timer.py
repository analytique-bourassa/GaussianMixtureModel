import time

class Timer:
    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        print('Elapsed time: %0.2fs' % (time.time() - self.t0))


    