# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:51:11 2016
@author: Ge Yang
"""

from MaskMaker import MaskMaker as MaskMaker, Utilities as mask_utils

import os
import subprocess
from time import sleep
from termcolor import colored, cprint

### Close DWG Viewer
try:
    subprocess.Popen(r'taskkill /F /im "dwgviewr.exe"')
except:
    pass

### initialization    
""" 
    first try to calculate the proper gapw for a certain impedance. 
    then calculate gapw for the resonator, which has a different centerpin 
    width
    """


### Userful Classes
class half_res():
    def __init__(self, s, freq, cap1, cap2, defaults={}):
        self.interior_length = calculate_interior_length(self.freq,
                                                         defaults['phase_velocity'],
                                                         defaults['impedance'],
                                                         resonator_type=0.5,
                                                         harmonic=0,
                                                         Ckin=cap1, Ckout=cap2)
        print("Interior length is: %f" % (self.interior_length))
        CPWWiggles(s, 5, self.interior_length / 2.)

    # for c in [c1, c2, c3]:
    ##TODO: Need to clean up the descriptions
    # c.short_description = lambda: "a"
    # c.long_description = lambda: "b"


    ### initialization
    """ 
    first try to calculate the proper gapw for a certain impedance. 
    then calculate gapw for the resonator, which has a different centerpin 
    width
    """


### Userful Classes

def set_mask_init():
    ### Setting defaults
    d = mask_utils.ChipDefaults()
    d.Q = 1000
    d.radius = 50
    d.segments = 6
    d.pinw_rsn = 2.  # this is the resonator pinwitdth that we are goign to use.
    d.gapw_rsn = 8.5

    d.pinw = 1.5  # d.pinw
    d.gapw = 1.
    d.center_gapw = 1
    ### Now calculate impedance
    d.imp_rsn = 80.  # calculate_impedance(d.pinw_rsn,d.gapw_rsn,d.eps_eff)
    d.solid = True
    return d


def chipInit(c, defaults):
    ### Components

    ### left and right launch points
    # experiment with different naming schemes for the launch points.
    left_and_right_launcher_spacing = 750

    launch_pt = MaskMaker.translate_pt(c.left_midpt, (0, left_and_right_launcher_spacing / 2.))
    setattr(c, 'sl1', MaskMaker.Structure(c, start=launch_pt, direction=0, defaults=defaults))
    launch_pt = MaskMaker.translate_pt(c.left_midpt, (0, - left_and_right_launcher_spacing / 2.))
    setattr(c, 'sl2', MaskMaker.Structure(c, start=launch_pt, direction=0, defaults=defaults))
    launch_pt = MaskMaker.translate_pt(c.right_midpt, (0, left_and_right_launcher_spacing / 2.))
    setattr(c, 'sr1', MaskMaker.Structure(c, start=launch_pt, direction=180, defaults=defaults))
    launch_pt = MaskMaker.translate_pt(c.right_midpt, (0, - left_and_right_launcher_spacing / 2.))
    setattr(c, 'sr2', MaskMaker.Structure(c, start=launch_pt, direction=180, defaults=defaults))

    ### Launchpoints
    # setattr(c, "d", defaults)
    setattr(c, 's1', MaskMaker.Structure(c, start=c.top_midpt, direction=270, defaults=defaults))
    setattr(c, 's2', MaskMaker.Structure(c, start=c.bottom_midpt, direction=90, defaults=defaults))
    setattr(c, 's3', MaskMaker.Structure(c, start=c.left_midpt, direction=0, defaults=defaults))
    setattr(c, 's4', MaskMaker.Structure(c, start=c.right_midpt, direction=180, defaults=defaults))
    setattr(c, 's5', MaskMaker.Structure(c, start=c.top_left, direction=270, defaults=defaults))
    setattr(c, 's6', MaskMaker.Structure(c, start=c.top_right, direction=270, defaults=defaults))
    setattr(c, 's7', MaskMaker.Structure(c, start=c.bottom_left, direction=90, defaults=defaults))
    setattr(c, 's8', MaskMaker.Structure(c, start=c.bottom_right, direction=90, defaults=defaults))
    # new anchor points
    setattr(c, 's9', MaskMaker.Structure(c, start=c.top_left_midpt, direction=270, defaults=defaults))
    setattr(c, 's10', MaskMaker.Structure(c, start=c.top_right_midpt, direction=270, defaults=defaults))
    setattr(c, 's11', MaskMaker.Structure(c, start=c.bottom_left_midpt, direction=90, defaults=defaults))
    setattr(c, 's12', MaskMaker.Structure(c, start=c.bottom_right_midpt, direction=90, defaults=defaults))
    # new anchor points
    setattr(c, 's13', MaskMaker.Structure(c, start=c.top_left_mid_left, direction=270, defaults=defaults))
    setattr(c, 's14', MaskMaker.Structure(c, start=c.top_left_mid_right, direction=270, defaults=defaults))
    setattr(c, 's15', MaskMaker.Structure(c, start=c.top_right_mid_left, direction=270, defaults=defaults))
    setattr(c, 's16', MaskMaker.Structure(c, start=c.top_right_mid_right, direction=270, defaults=defaults))
    setattr(c, 's17', MaskMaker.Structure(c, start=c.bottom_left_mid_left, direction=90, defaults=defaults))
    setattr(c, 's18', MaskMaker.Structure(c, start=c.bottom_left_mid_right, direction=90, defaults=defaults))
    setattr(c, 's19', MaskMaker.Structure(c, start=c.bottom_right_mid_left, direction=90, defaults=defaults))
    setattr(c, 's20', MaskMaker.Structure(c, start=c.bottom_right_mid_right, direction=90, defaults=defaults))
    MaskMaker.FineAlign(c)


def chipDrw_1(c, chip_resonator_length, chip_coupler_length, d=None):
    ### Chip Init
    # gapw=calculate_gap_width(eps_eff,50,pinw)

    print(d.__dict__.keys())
    if d == None: d = MaskMaker.ChipDefaults()
    gap_ratio = d.gapw / d.pinw
    c.frequency = 10
    c.pinw = 1.5  # d.pinw
    c.gapw = 1.
    c.center_gapw = 1
    c.radius = 40
    c.frequency = 5  # GHz
    # c.Q = 1000000 #one of the couplers
    c.imp_rsn = 80
    launch_pin_width = 150
    launcher_width = (1 + 2 * gap_ratio) * launch_pin_width
    c.launcher_length = 500
    c.inside_padding = 1160
    c.left_inside_padding = 180
    c.C = 0.5e-15  # (factor of 2 smaller than actual value from simulation.)
    c.cap1 = mask_utils.sapphire_capacitor_by_C_Channels(c.C)
    # c.cap1 = sapphire_capacitor_by_Q_Channels(c.frequency, c.Q, 50, resonator_type=0.5)
    ### Calculating the interior length of the resonator
    # c.interior_length = calculate_interior_length(c.frequency, d.phase_velocity,
    # c.imp_rsn, resonator_type=0.5,
    # harmonic=0, Ckin=c.cap1, Ckout=c.cap1)
    c.interior_length = 17631 / 2.  # unit is micron, devide by 2 to get the half wavelength
    # This gives 6.88GHz in reality.
    c.meander_length = c.interior_length  # (c.interior_length - (c.inside_padding))  - c.left_inside_padding
    print('meander_length', c.meander_length)

    c.num_wiggles = 7
    c.meander_horizontal_length = (c.num_wiggles + 1) * 2 * c.radius
    launcher_straight = 645
    chipInit(c, defaults=d)

    ### Components
    #### MaskMaker.Launcher
    two_layer = c.two_layer

    c.s5.pinw = 4.
    c.s5.gapw = 2.5
    c.s7.pinw = 4.
    c.s7.gapw = 2.5
    c.s3.pinw2 = d.pinw_rsn
    c.s4.pinw2 = d.pinw_rsn
    if two_layer:
        c.s5.chip.two_layer = True
        c.s7.chip.two_layer = True

    # resonator_length = 3000  # for 10GHz

    length_before_turning = 20  # 220 + 116
    turn_radius = 100
    turn_radius_2 = 10

    coupler_width_1 = 5
    length_2 = 750 / 2. - turn_radius - turn_radius_2 - coupler_width_1
    ### Left launchers and the drive resonator
    s = c.sl1
    MaskMaker.Launcher(s, pinw=2, gapw=2)
    MaskMaker.CPWStraight(s, length_before_turning)
    MaskMaker.CPWBend(s, -90, radius=turn_radius, segments=10)
    MaskMaker.CPWStraight(s, length_2)
    MaskMaker.CPWBend(s, 90, radius=turn_radius_2, segments=10)
    # MaskMaker.CPWTaper(s, 10, 0.09, stop_pinw=1, stop_gapw=.5)

    # Interaction driven programming: encourage trying code out and look at the output, instead of just reading code.
    # such encouragements unlock new behaviors that positively re-enforce itself.
    # this positive re-enforcement is what makes this kind of philosophical change significant.

    s = c.sl2
    MaskMaker.Launcher(s, pinw=2, gapw=2)
    MaskMaker.CPWStraight(s, length_before_turning)
    MaskMaker.CPWBend(s, 90, radius=turn_radius, segments=10)
    MaskMaker.CPWStraight(s, length_2)
    MaskMaker.CPWBend(s, -90, radius=turn_radius_2, segments=10)
    # MaskMaker.CPWTaper(s, 10, 0.09, stop_pinw=1, stop_gapw=.5)

    # c.sl1.last = MaskMaker.middle(c.sl1.last, c.sl2.last)
    # CPWWiggles(c.sl1, 1, 12, 1, start_up=True, radius=(1.5 + 2) / 2., segments=10)

    # drive resonator from the left
    drive_resonator_start_pt = MaskMaker.middle(c.sl1.last, c.sl2.last)
    c.s_drive = MaskMaker.Structure(c, start=drive_resonator_start_pt, direction=0)
    MaskMaker.CoupledTaper(c.s_drive, 6, pinw=c.sl1.pinw, gapw=c.sl1.gapw, center_gapw=8, stop_pinw=1.2, stop_gapw=1,
                           stop_center_gapw=1)
    MaskMaker.CoupledStraight(c.s_drive, 50)
    MaskMaker.CoupledStraight(c.s_drive, 2500 + 302.910 - 300)

    MaskMaker.CoupledTaper(c.s_drive, 4, stop_center_gapw=0.0, stop_pinw=0.16 + 0.110, stop_gapw=0.220)
    MaskMaker.CoupledStraight(c.s_drive, 7.09)
    c.s_drive.last = MaskMaker.translate_pt(c.s_drive.last, (0, .19 - 0.0550))
    MaskMaker.CPWBend(c.s_drive, -180, radius=0.19 - 0.055, segments=7)

    ### Right launchers and the readout resonator

    s = c.s4
    s.chip.two_layer = True
    s.pinw = 1.2
    s.gapw = 1
    MaskMaker.Inductive_Launcher(s, pinw=25, gapw=20, padw=200, padl=300, num_loops=4)
    MaskMaker.CPWTaper(s, 80, stop_pinw=1.5, stop_gapw=1.5)
    MaskMaker.CPWStraight(s, 3000 - 80 - 245. - chip_resonator_length + 280 + 229.890 + 300)
    MaskMaker.CPWTaper(s, 1, stop_pinw=.380, stop_gapw=.210)
    MaskMaker.CPWStraight(s, 4.5)
    MaskMaker.CPWTaper(s, 1, stop_pinw=1.5, stop_gapw=1.5)
    MaskMaker.CPWStraight(s, 10)
    MaskMaker.CPWTaper(s, 3, stop_pinw=3.4, stop_gapw=1)

    # readout resonator on the right
    readout_resonator_start_pt = MaskMaker.translate_pt(c.center, (chip_resonator_length - 529.390 - 300, 0))
    c.s_readout = MaskMaker.Structure(c, start=readout_resonator_start_pt, direction=180)
    s = c.s_readout
    s.pinw = 1.2
    s.gapw = 1
    s.center_gapw = 1
    MaskMaker.CoupledStraight(s, 50)
    MaskMaker.CoupledWiggles(s, 6, 1000, 0, start_up=True, radius=30,
                             segments=10)
    MaskMaker.CoupledStraight(s, resonator_length - 1000 - 1 - 1.16)
    MaskMaker.CoupledTaper(s, 1, stop_center_gapw=0.22, stop_pinw=0.16, stop_gapw=0.220 + 2.210 - 1.5)
    MaskMaker.CoupledStraight(s, 1.16, center_gapw=0.22, pinw=0.16, gapw=0.220)

    # if not hasattr(s, 'gap_layer') and not hasattr(s, 'pin_layer'):
    #     MaskMaker.CPWStraight(s, .1225, pinw=0, gapw=.22 / 2.)

    # DC bias lead
    length_to_trap = 450 - 10
    guard_middle_pinch_length = 5
    guardSHorizontal = 445. - 300

    s = c.s14
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, length_to_trap - 400)
    MaskMaker.CPWSturn(s, 20, 90, 90, guardSHorizontal, -90, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, 180.51)
    MaskMaker.CPWTaper(s, 4, stop_pinw=1.22, stop_gapw=0.180)
    MaskMaker.CPWStraight(s, guard_middle_pinch_length)

    s = c.s18
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, length_to_trap - 400)
    MaskMaker.CPWSturn(s, 20, -90, 90, guardSHorizontal, 90, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, 180.51)
    MaskMaker.CPWTaper(s, 4, stop_pinw=1.22, stop_gapw=0.180)
    MaskMaker.CPWStraight(s, guard_middle_pinch_length)

    mid_pt = MaskMaker.translate_pt(MaskMaker.middle(c.s1.last, c.s2.last), (-300, 0))
    s.last = mid_pt
    MaskMaker.Elipses(s, mid_pt, 0.85, 1.4, 0, 20)
    MaskMaker.Elipses(s, mid_pt, 0.85 - 0.2, 1.4 - 0.2, 0, 20)

    s.chip.two_layer = True

    middleGuardL = 1.2
    sideGuardL = 0.2
    guardGap = 0.2

    guardLauncherStraight = 145 + 1.3759 + 0.3750
    guardSHorizontal = 360 + 12.68 + 1.277 + 0.0707 - 0.220 - 0.0670 - 0.290
    guardEndStraight = 120 + 20.7
    sideGuardEndStraight = 1.0 + 3.9897 + 0.255 - 0.12 - 0.25

    ### Microwave Feed Couplers

    R1 = 100
    L1 = 0
    L2 = 300 - 80 - 40 - 40
    L3 = 199.5 - 0.0013 - 120 - 40 - 40
    L4 = 100
    L5 = 500 - 386.6025 - 40

    s = c.s15
    s.chip.two_layer = False
    s.pinw = 3
    s.gapw = 1
    MaskMaker.Launcher(s)
    MaskMaker.CPWSturn(s, L1, -90, R1, L2, 90, R1, L3, segments=5)
    MaskMaker.CPWBend(s, -60, radius=100, segments=12)
    MaskMaker.CPWStraight(s, L4)
    MaskMaker.CPWBend(s, -30, radius=50, segments=12)
    MaskMaker.CPWStraight(s, chip_coupler_length + L5)
    MaskMaker.CPWStraight(s, 1.5, pinw=0, gapw=2.5)

    s = c.s19
    s.chip.two_layer = False
    s.pinw = 3
    s.gapw = 1
    MaskMaker.Launcher(s)
    MaskMaker.CPWSturn(s, L1, 90, R1, L2, -90, R1, L3, segments=5)
    MaskMaker.CPWBend(s, 60, radius=100, segments=12)
    MaskMaker.CPWStraight(s, L4)
    MaskMaker.CPWBend(s, 30, radius=50, segments=12)
    MaskMaker.CPWStraight(s, chip_coupler_length + L5)
    MaskMaker.CPWStraight(s, 1.5, pinw=0, gapw=2.5)

    ### Resonator Couplers
    c.two_layer = True
    coupler_offset_v = 1.5 + 1.2 - 1.
    coupler_offset_h = 100
    launch_pt = MaskMaker.translate_pt(c.center, (coupler_offset_h, coupler_offset_v))
    setattr(c, 'resonator_coupler_1', MaskMaker.Structure(c, start=launch_pt, direction=90))
    launch_pt = MaskMaker.translate_pt(c.center, (coupler_offset_h, -coupler_offset_v))
    setattr(c, 'resonator_coupler_2', MaskMaker.Structure(c, start=launch_pt, direction=-90))

    coupler_1 = c.resonator_coupler_1
    coupler_1.pinw = 1.5
    coupler_1.gapw = 1.5
    coupler_2 = c.resonator_coupler_2
    coupler_2.pinw = 1.5
    coupler_2.gapw = 1.5

    MaskMaker.CPWStraight(coupler_1, 50 - 0.75)
    MaskMaker.CPWBend(coupler_1, 90, radius=50, segments=12)
    MaskMaker.CPWStraight(coupler_1, chip_coupler_length)
    MaskMaker.CPWStraight(coupler_1, 1.5, pinw=0, gapw=4.5 / 2.)

    MaskMaker.CPWStraight(coupler_2, 50 - 0.75)
    MaskMaker.CPWBend(coupler_2, -90, radius=50, segments=12)
    MaskMaker.CPWStraight(coupler_2, chip_coupler_length)
    MaskMaker.CPWStraight(coupler_2, 1.5, pinw=0, gapw=4.5 / 2.)

    ### DC Guards
    guard_pin_w = 0.8

    s = c.s5
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, guardLauncherStraight)
    MaskMaker.CPWSturn(s, 20, 90, 90, guardSHorizontal + 1875 - 300, -60, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, guardEndStraight)
    MaskMaker.CPWTaper(s, 4, stop_pinw=guard_pin_w, stop_gapw=0.180)
    MaskMaker.CPWBend(s, -30, radius=0.5, segments=2)
    MaskMaker.CPWStraight(s, sideGuardEndStraight)

    s = c.s7
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, guardLauncherStraight)
    MaskMaker.CPWSturn(s, 20, -90, 90, guardSHorizontal + 1875 - 300, 60, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, guardEndStraight)
    MaskMaker.CPWTaper(s, 4, stop_pinw=guard_pin_w, stop_gapw=0.180)
    MaskMaker.CPWBend(s, 30, radius=0.5, segments=2)
    MaskMaker.CPWStraight(s, sideGuardEndStraight)

    # Bias DC Pinch DC Electrodes
    L1 = 20
    R = 20
    L3 = - (resonator_length - 1900) + 1450 - 40 - 36.86

    L5 = 5

    L4 = 450 - 40 - L1 - 0.4 - L5

    s = c.s6
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWSturn(s, L1, -90, R, L3, 90, R, L4, segments=3)
    MaskMaker.CPWTaper(s, L5, stop_pinw=4, stop_gapw=0.25)

    s = c.s8
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWSturn(s, L1, 90, R, L3, -90, R, L4, segments=3)
    MaskMaker.CPWTaper(s, L5, stop_pinw=4, stop_gapw=0.25)

    # Resonator Side Guards
    s = c.s1
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, guardLauncherStraight)
    MaskMaker.CPWSturn(s, 20, -90, 90, guardSHorizontal - 325, 60, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, guardEndStraight)
    MaskMaker.CPWTaper(s, 4, stop_pinw=guard_pin_w, stop_gapw=0.180)
    MaskMaker.CPWBend(s, 30, radius=0.5, segments=2)
    MaskMaker.CPWStraight(s, sideGuardEndStraight)

    s = c.s2
    s.chip.two_layer = False
    MaskMaker.Launcher(s, pinw=1.5, gapw=1.5)
    MaskMaker.CPWStraight(s, guardLauncherStraight)
    MaskMaker.CPWSturn(s, 20, 90, 90, guardSHorizontal - 325, -60, 90, 20, segments=10)
    MaskMaker.CPWStraight(s, guardEndStraight)
    MaskMaker.CPWTaper(s, 4, stop_pinw=guard_pin_w, stop_gapw=0.180)
    MaskMaker.CPWBend(s, -30, radius=0.5, segments=2)
    MaskMaker.CPWStraight(s, sideGuardEndStraight)

    s.chip.two_layer = True


if __name__ == "__main__":
    ### define mask name, and open up an explorer window
    MaskName = "M017V5"  # M006 Constriction Gate Resonator"

    m = MaskMaker.WaferMask(MaskName, flat_angle=90., flat_distance=24100., wafer_padding=3.3e3, chip_size=(7000, 1900),
                            dicing_border=400, etchtype=False, wafer_edge=False,
                            dashed_dicing_border=50)
    print("chip size: ", m.chip_size)
    # Smaller alignment markers as requested by Leo
    points = [  # (-11025., -19125.),(-11025., 19125.),(11025., -19125.),(11025., 19125.),
        (-15000., -13200.), (-15000., 13200.), (15000., -13200.), (15000., 13200.)]
    MaskMaker.AlignmentCross(m, linewidth=10, size=(1000, 1000), points=points, layer='gap', name='cross')
    border_locs = [(-18750., 21600.), (18750., 21600.),
                   (-18750., -21600.), (18750., -21600.)]
    MaskMaker.AlignmentCross(m, linewidth=200, size=(200, 200), points=border_locs, layer='gap', name='border_gap')
    MaskMaker.AlignmentCross(m, linewidth=200, size=(200, 200), points=border_locs, layer='pin', name='border_pin')
    MaskMaker.AlphaNumText(m, text="DIS 160923", size=(400, 400), point=(7650, -19300), centered=True,
                           layer='gap')  # change the mask name/title here
    MaskMaker.AlphaNumText(m, text="M017 Ge Yang", size=(400, 400), point=(7650, -18500), centered=True,
                           layer='gap')  # change the mask name/title here
    MaskMaker.AlphaNumText(m, text="DIS 160923 M017 Ge Yang", size=(700, 700), point=(0, 20500), centered=True,
                           layer='gap')  # change the mask name/title here

    # m.randomize_layout()
    d = set_mask_init()
    # two_layer = True
    # solid = False
    two_layer = True
    solid = True

    for i, resonator_length in enumerate([2440, 2140, 1900]):
        for j, coupler_length in enumerate([50, 45, 40]):
            chip_name = 'YggdrasilV1r{}c{}'.format(i + 1, j + 1)
            print('chip name: ', chip_name)
            c = MaskMaker.Chip(chip_name, author='GeYang', size=m.chip_size, mask_id_loc=(5050, 1720),
                               chip_id_loc=(4540, 100), two_layer=two_layer, solid=solid, segments=10)
            c.textsize = (80, 80)
            chipDrw_1(c, chip_resonator_length=resonator_length, chip_coupler_length=coupler_length, d=d)
            m.add_chip(c, 1)

    m.save()
    ### Check and open the folder
    cprint(
        colored('current directory ', 'grey') +
        colored(os.getcwd(), 'green')
    )
    sleep(.1)

    cprint(
        colored("operating system is: ", 'grey') +
        colored(os.name, 'green')
    )
    chip_path = os.path.join(os.getcwd(), MaskName + '-' + c.name + '.dxf')
    if os.name == 'posix':
        subprocess.call('open -a "AutoCAD 2016" ' + chip_path, shell=True)
    elif os.name == 'nt':
        subprocess.Popen(
            r'"C:\Program Files\Autodesk\DWG TrueView 2014\dwgviewr.exe" "' + chip_path + '" ')
