from vmo.VMO.oracle import build_oracle
import unittest

SYMBOLIC_SEQ = ['a', 'b' , 'b', 'c', 'a', 'b', 'c', 'd', 'a', 'b', 'c']

class VmoTest(unittest.TestCase):

    def build_oracle_test(self):
        p = build_oracle(SYMBOLIC_SEQ, 'f')
        self.assertEqual([0, 0, 0, 1, 0, 1, 2, 2, 0, 1, 2, 3], p.lrs)
        self.assertEqual([None, 0, 0, 2, 0, 1, 2, 4, 0, 1, 2, 7], p.sfx)


if __name__ == '__main__':
    unittest.main()