import unittest
from PyParticle.viz import make_grid, plot_lines, format_axes, add_legend

class TestViz(unittest.TestCase):

    def test_make_grid(self):
        # Test for a simple grid layout
        fig, axarr = make_grid(2, 2, figsize=(10, 8))
        self.assertEqual(axarr.shape, (2, 2))
        self.assertEqual(fig.get_size_inches().tolist(), [10, 8])

    def test_plot_lines(self):
        # Mock data for testing plot_lines
        mock_population = [1, 2, 3]  # Replace with actual mock population data
        fig, ax = plt.subplots()
        line, labs = plot_lines("dNdlnD", (mock_population,), var_cfg=None, ax=ax)
        self.assertIsNotNone(line)
        self.assertIsInstance(labs, (list, tuple))

    def test_format_axes(self):
        fig, ax = plt.subplots()
        format_axes(ax, xlabel='X-axis', ylabel='Y-axis', title='Test Title')
        self.assertEqual(ax.get_xlabel(), 'X-axis')
        self.assertEqual(ax.get_ylabel(), 'Y-axis')
        self.assertEqual(ax.get_title(), 'Test Title')

    def test_add_legend(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9], label='Test Line')
        add_legend(ax)
        self.assertIsNotNone(ax.get_legend())

if __name__ == '__main__':
    unittest.main()