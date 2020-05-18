import unittest
from slam.main import main

class LocalMapManagerMethods(unittest.TestCase):
    def test_tracking(self):
        main(flag_use_camera=False)

if __name__ == "__main__":
    unittest.main()