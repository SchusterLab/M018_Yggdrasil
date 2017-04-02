from MaskMaker import MaskMaker as mm, Utilities as mask_utils, sdxf


def shunted_launcher(structure, pinw, gapw, handedness='right'):
    """A very specific shunted launcher component"""
    offset = 34  # to be flush with other launchers
    extra_offset = 50
    l1 = 4
    mm.CPWStraight(structure, offset + extra_offset, pinw=0, gapw=0)
    mm.Inductive_Launcher(structure, pinw, gapw, padw=150, padl=200, num_loops=7,
                          handedness=handedness)
    mm.CPWStraight(structure, l1)
    mm.CPWCapacitiveShunt(structure, 20, 140, finger_gapw=2, finger_pinw=4)
    mm.CPWStraight(structure, 2)


def girl_with_bonding_tatoo(structure, handedness='right'):
    pinw = structure.pinw
    padding = 20
    width = 150
    length = 200
    mm.CPWStraight(structure, padding, pinw=pinw, gapw=padding + (width - pinw) / 2)
    mm.CPWStraight(structure, length, pinw=width, gapw=0)
    mm.CPWStraight(structure, padding, pinw=0, gapw=padding + width / 2)
    mm.CPWStraight(structure, -padding, pinw=0, gapw=0)
    mm.CPWStraight(structure, -length / 2, pinw=0, gapw=0)
    if handedness == 'right':
        structure.last_direction += 90
    elif handedness == 'left':
        structure.last_direction -= 90
    mm.CPWStraight(structure, - width / 2 - padding, pinw=0, gapw=0)
    mm.CPWStraight(structure, padding, pinw=0, gapw=length / 2)
    mm.CPWStraight(structure, width, pinw=0, gapw=0)
    mm.CPWStraight(structure, padding, pinw=pinw, gapw=(length - pinw) / 2)


def bend_and_touch_down(structure, handedness, guardEndStraight, guard_taper_length, guard_pin_w, trap_gap,
                        guard_radius, sideGuardEndStraight):

    r = 30
    l = 70 + 0.3494 + 0.1793
    if handedness == 'right':
        a1 = -60
        a2 = -30
    elif handedness == 'left':
        a1 = 60
        a2 = 30
    mm.CPWBend(structure, a1, radius=r, segments=10)
    mm.CPWStraight(structure, l)

    mm.CPWStraight(structure, guardEndStraight)
    mm.CPWTaper(structure, guard_taper_length, stop_pinw=guard_pin_w, stop_gapw=trap_gap)
    mm.CPWBend(structure, a2, radius=guard_radius, segments=1)
    mm.CPWStraight(structure, sideGuardEndStraight)
