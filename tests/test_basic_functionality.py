#!/usr/bin/env python3
"""
Basic functionality tests for AirSim RACER
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exploration.grid_map import GridMap
from exploration.frontier_finder import FrontierFinder
from planning.path_planner import PathPlanner
from utils.math_utils import distance_3d, normalize_vector


class TestGridMap(unittest.TestCase):
    def setUp(self):
        self.bounds_min = np.array([-10.0, -10.0, 0.0])
        self.bounds_max = np.array([10.0, 10.0, 5.0])
        self.resolution = 0.5
        self.grid_map = GridMap(self.bounds_min, self.bounds_max, self.resolution)
    
    def test_initialization(self):
        """Test grid map initialization"""
        self.assertEqual(self.grid_map.size_x, 40)  # (20 / 0.5)
        self.assertEqual(self.grid_map.size_y, 40)
        self.assertEqual(self.grid_map.size_z, 10)  # (5 / 0.5)
    
    def test_coordinate_conversion(self):
        """Test world to grid coordinate conversion"""
        world_pos = np.array([0.0, 0.0, 2.5])
        grid_pos = self.grid_map.world_to_grid(world_pos)
        
        # Should be at center of grid
        self.assertEqual(grid_pos[0], 20)
        self.assertEqual(grid_pos[1], 20)
        self.assertEqual(grid_pos[2], 5)
        
        # Test conversion back
        world_pos_back = self.grid_map.grid_to_world(grid_pos)
        np.testing.assert_array_almost_equal(world_pos, world_pos_back, decimal=1)
    
    def test_occupancy_queries(self):
        """Test occupancy queries"""
        test_pos = np.array([0.0, 0.0, 2.5])
        
        # Initially should be unknown
        self.assertTrue(self.grid_map.is_unknown(test_pos))
        self.assertFalse(self.grid_map.is_occupied(test_pos))
        self.assertFalse(self.grid_map.is_free(test_pos))


class TestMathUtils(unittest.TestCase):
    def test_distance_3d(self):
        """Test 3D distance calculation"""
        p1 = (0, 0, 0)
        p2 = (3, 4, 0)
        dist = distance_3d(p1, p2)
        self.assertAlmostEqual(dist, 5.0, places=5)
    
    def test_normalize_vector(self):
        """Test vector normalization"""
        vector = np.array([3.0, 4.0, 0.0])
        normalized = normalize_vector(vector)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected, decimal=5)
        
        # Test magnitude is 1
        magnitude = np.linalg.norm(normalized)
        self.assertAlmostEqual(magnitude, 1.0, places=5)


class TestFrontierFinder(unittest.TestCase):
    def setUp(self):
        bounds_min = np.array([-10.0, -10.0, 0.0])
        bounds_max = np.array([10.0, 10.0, 5.0])
        self.grid_map = GridMap(bounds_min, bounds_max, 0.5)
        self.frontier_finder = FrontierFinder(self.grid_map)
    
    def test_initialization(self):
        """Test frontier finder initialization"""
        self.assertIsNotNone(self.frontier_finder.grid_map)
        self.assertEqual(self.frontier_finder.cluster_tolerance, 1.0)
        self.assertEqual(self.frontier_finder.min_frontier_size, 5)


class TestPathPlanner(unittest.TestCase):
    def setUp(self):
        bounds_min = np.array([-10.0, -10.0, 0.0])
        bounds_max = np.array([10.0, 10.0, 5.0])
        self.grid_map = GridMap(bounds_min, bounds_max, 0.5)
        self.path_planner = PathPlanner(self.grid_map)
    
    def test_initialization(self):
        """Test path planner initialization"""
        self.assertIsNotNone(self.path_planner.grid_map)
        self.assertEqual(self.path_planner.inflation_radius, 0.5)


class TestIntegration(unittest.TestCase):
    def test_import_all_modules(self):
        """Test that all modules can be imported without errors"""
        try:
            from exploration.exploration_manager import ExplorationManager, ExplorationConfig
            from airsim_interface.airsim_client import AirSimClient
            from exploration.grid_map import GridMap
            from exploration.frontier_finder import FrontierFinder
            from planning.path_planner import PathPlanner
            from utils.math_utils import distance_3d, normalize_vector
        except ImportError as e:
            self.fail(f"Import failed: {e}")


def run_basic_tests():
    """Run basic functionality tests"""
    print("Running AirSim RACER basic functionality tests...")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(loader.loadTestsFromTestCase(TestGridMap))
    suite.addTest(loader.loadTestsFromTestCase(TestMathUtils))
    suite.addTest(loader.loadTestsFromTestCase(TestFrontierFinder))
    suite.addTest(loader.loadTestsFromTestCase(TestPathPlanner))
    suite.addTest(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {len(result.failures)} failures, {len(result.errors)} errors")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(run_basic_tests())