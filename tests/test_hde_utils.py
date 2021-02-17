import hde_utils as utl

spike_times = utl.get_spike_times_from_file('sample_data/spike_times.dat')


def test_get_spike_times_from_file():
    spike_times = utl.get_spike_times_from_file('sample_data/spike_times.dat')
    assert len(spike_times) > 0
