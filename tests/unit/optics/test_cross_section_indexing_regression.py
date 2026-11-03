import numpy as np
from part2pop.optics.base import OpticalParticle
from part2pop.aerosol_particle import make_particle


class Dummy(OpticalParticle):
    def compute_optics(self):
        pass


def test_simultaneous_and_individual_axis_selection():
    p = make_particle(100e-9, ['SO4'], [1.])
    o = Dummy(p, {'rh_grid':[0., .5], 'wvl_grid':[500e-9, 600e-9]})
    o.Cext[:] = [[1., 2.], [3., 4.]]
    assert o.get_cross_section('ext', rh_idx=1, wvl_idx=0) == 3.
    np.testing.assert_array_equal(o.get_cross_section('ext', rh_idx=1), [3., 4.])
    np.testing.assert_array_equal(o.get_cross_section('ext', wvl_idx=0), [1., 3.])


def test_homogeneous_particle_single_axis_selection():
    from part2pop.optics.factory.homogeneous import HomogeneousParticle
    particle = object.__new__(HomogeneousParticle)
    particle.Cext = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_array_equal(particle.get_cross_section('ext', wvl_idx=0), [1., 3.])
    np.testing.assert_array_equal(particle.get_cross_section('ext', rh_idx=1), [3., 4.])
