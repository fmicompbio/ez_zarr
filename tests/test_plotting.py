# for testing, run the following from the project folder:
#     pip install -e .
#     pytest --color=yes -v --cov=./src --cov-report=term-missing tests

import pytest
import numpy as np
import dask.array
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import warnings

from ez_zarr.plotting import zproject, get_shuffled_cmap, pad_image, convert_to_rgb, plot_image


# fixtures --------------------------------------------------------------------
@pytest.fixture
def npa3d() -> np.ndarray:
    """A 3D `numpy.array` with shape (3,4,5)"""
    np.random.seed(42)
    return np.random.randint(0, 16, size=(3, 4, 5))

@pytest.fixture
def npa4d() -> np.ndarray:
    """A 4D `numpy.array` with shape (2,3,4,5)"""
    np.random.seed(42)
    return np.random.randint(0, 2**16, size=(2, 3, 4, 5))

# global variables ------------------------------------------------------------
def test_get_shuffled_cmap():
    """Test get_shuffled_cmap."""
    cm = get_shuffled_cmap()
    assert isinstance(cm, mcolors.ListedColormap)

# helper functions ------------------------------------------------------------
def test_zproject(npa3d: np.ndarray, npa4d: np.ndarray):
    """Test zproject."""
    with pytest.raises(Exception) as e_info:
        zproject(im=npa3d, axis=1, method='error')

    r3d0max = zproject(im=npa3d, axis=0, method='maximum',
                       keepdims=True, img_bit=16)
    r3d0min = zproject(im=npa3d, axis=0, method='minimum',
                    keepdims=True, img_bit=16)
    r3d0sum = zproject(im=npa3d, axis=0, method='sum',
                       keepdims=True, img_bit=16)
    r3d0avg = zproject(im=npa3d, axis=0, method='average',
                       keepdims=True, img_bit=16)
    r4d1sum_noclip = zproject(im=npa4d, axis=1, method='sum',
                              keepdims=True, img_bit=32)
    r4d1avg_reddim = zproject(im=npa4d, axis=1, method='average',
                              keepdims=False, img_bit=16)
    assert isinstance(r3d0max, np.ndarray)
    assert r3d0max.shape == (1,4,5)
    assert r3d0min.shape == r3d0max.shape
    assert r3d0sum.shape == r3d0max.shape
    assert r3d0avg.shape == r3d0max.shape
    assert r4d1sum_noclip.shape == (2,1,4,5)
    assert r4d1avg_reddim.shape == (2,4,5)
    assert (r3d0max == np.max(npa3d, axis=0, keepdims=True)).all()
    assert (r3d0min == np.min(npa3d, axis=0, keepdims=True)).all()
    assert (r3d0sum == np.clip(np.sum(npa3d, axis=0, keepdims=True), 0, 2**16)).all()
    assert (r3d0avg == np.mean(npa3d, axis=0, keepdims=True)).all()
    assert (r4d1sum_noclip == np.sum(npa4d, axis=1, keepdims=True)).all()
    assert (r4d1avg_reddim == np.mean(npa4d, axis=1, keepdims=False)).all()

def test_pad_image(npa3d: np.ndarray):
    """Test pad_image."""
    pad = (0, 20, 30)
    out = tuple(pad[i] + npa3d.shape[i] for i in range(3))
    impad = pad_image(im = npa3d, output_shape=out)
    assert impad.shape == out
    assert (impad[:, slice(10, 10 + npa3d.shape[1]), slice(15, 15 + npa3d.shape[2])] == npa3d).all()

def test_convert_to_rgb(npa3d: np.ndarray):
    """Test convert_to_rgb."""
    rng = np.quantile(npa3d[1], [0.01, 0.5])
    rgb = convert_to_rgb(im=npa3d[[0,1]],
                         channel_colors=['yellow', 'red'],
                         channel_ranges=[[0.01, 0.5], rng])
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (npa3d.shape[1], npa3d.shape[2], 3)
    assert rgb.dtype == np.uint8
    assert np.max(rgb) == 255
    da = dask.array.from_array(npa3d)
    rgb = convert_to_rgb(im=da[[0,1]],
                         channel_colors=['yellow', 'red'],
                         channel_ranges=[[0.01, 0.5], rng])
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (npa3d.shape[1], npa3d.shape[2], 3)
    assert rgb.dtype == np.uint8
    assert np.max(rgb) == 255

def test_plot_image(npa4d: np.ndarray, npa3d: np.ndarray, tmpdir: str):
    """Test plot_image."""
    with pytest.raises(Exception) as e_info:
        plot_image(im=npa4d, msk=None,
                   channels=[1],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   title='test', call_show=False,
                   scalebar_pixel=100,
                   scalebar_position='error')

    with pytest.raises(Exception) as e_info:
        plot_image(im=npa4d, msk=None,
                   channels=[0, 1],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   title='test')

    with pytest.raises(Exception) as e_info:
        plot_image(im=npa4d, msk=None,
                   channels=[0, 1],
                   channel_colors=['viridis', 'inferno'],
                   channel_ranges=[[0.01, 0.99]],
                   title='test')

    with pytest.raises(Exception) as e_info:
        plot_image(im=npa4d, msk=None,
                   channels=[1],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   title='test', call_show=False,
                   axis_style='error')

    matplotlib.use('Agg')  # Use the 'Agg' backend, which doesn't need a display
    with warnings.catch_warnings():
        warnings.simplefilter('ignore') # suppress warning due to cannot show FigureCanvas 
        # scalebar_position
        for pos in ['topleft', 'bottomleft', 'topright', 'bottomright']:
            plot_image(im=npa4d, msk=None,
                    channels=[1],
                    channel_colors=['white'],
                    channel_ranges=[[0.01, 0.99]],
                    title='test', title_fontsize='xx-large', call_show=False,
                    scalebar_pixel=100, scalebar_fontsize=10,
                    scalebar_position=pos,
                    scalebar_label='test')
        # axis_style and image_transform
        for st in ['none', 'pixel', 'frame', 'micrometer']:
            plot_image(im=npa4d, msk=None,
                       channels=[1],
                       channel_colors=['white'],
                       channel_ranges=[[0.01, 0.99]],
                       title='test', call_show=False,
                       image_transform=np.log1p,
                       axis_style=st,
                       spacing_yx=[0.325, 0.325])
        # single channel with colormap
        plot_image(im=npa4d, msk=None,
                    channels=[1],
                    channel_colors=['inferno'],
                    channel_ranges=[[0.01, 0.99]],
                    title='test', call_show=False)
        # image with masks
        plot_image(im=npa4d, msk=npa3d,
                   channels=[1],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   title='test', call_show=True)
        plot_image(im=npa4d, msk=npa3d,
                   channels=[1],
                   channel_colors=['white'],
                   channel_ranges=[[0.01, 0.99]],
                   show_label_values=True,
                   label_text_colour='red', label_fontsize=12,
                   title='test', call_show=True)
        # brightfield image
        plot_image(im=npa4d[slice(0,1)],
                   call_show=True,
                   fig_style='brightfield')
        # 2D image
        plot_image(im=npa4d[0, 0],
                   call_show=True)
    plt.savefig(tmpdir.join('output.png'))
    assert True # check if the plotting ran through

