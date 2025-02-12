import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
NAME
    mtf_curve  

PURPOSE
    Simulate the Modulation Transfer Function (MTF) for film, lenses, scanners, and sharpening.

CALLING SEQUENCE
    mtf_curve(f50, fboost=0, flens=0, lord=2, dscan=0, sincpwr=0, ksharp=0)

RETURN
    Spatial frequency array and total MTF values.

RESTRICTIONS
    - f50 must be > 0 for film MTF computation.
    - fboost must be < 0.45 * f50.
    - MTF_lens must be between 0 and 1 (exclusive) when estimating flens.

EXAMPLE
    mtf_curve(f50)                   # f50 is the 50% MTF density of film in line pairs/mm, typically 20 - 60.
    mtf_curve(f50, fboost)           # fboost is the frequency shift of the 2nd order equalizer.
    mtf_curve(f50, fboost, flens)    # flens is the 50% MTF density of the lens.
    mtf_curve(f50, fboost, flens, lord)  # lord is lens rolloff order (default = 2).
    mtf_curve(f50, fboost, flens, lord, dscan, sincpwr)  # dscan is scanner pixels per mm (dpmm).
                                    # sincpwr is the power of the sinc function describing the scanner or digital sensor rolloff.
    mtf_curve(f50, fboost, flens, lord, dscan, sincpwr, ksharp)  # ksharp is sharpening boost.

LITERATURE
    https://www.normankoren.com/Tutorials/MTF1A.html

HISTORY
    Written   Normal Koren,   Mar   2007  : Created in MatLab
    Rewritten Luis Samaniego, Feb   2025  : Rewritten into Python, support by ChatGPT
    Modified  Luis Samaniego, Feb   2025  : Added flens function, dynamic legend, f50 values

Copyright (C) 2001-2004 by Norman Koren
http://www.normankoren.com  normkoren@earthlink.net
Every conceivable disclaimer applies; the author assumes no legal liability whatsoever.
You may distribute this program freely, but please maintain the original copyright.
"""

def sinc(x):
    """Sinc function: sin(pi*x)/(pi*x)"""
    return np.sinc(x / np.pi)

def find_f50(f, mtf):
    """Finds the spatial frequency where MTF = 0.5"""
    return np.interp(0.5, np.flip(mtf), np.flip(f))

def estimate_flens(MTF_lens, f, lord=2):
    """
    Estimate flens (50% MTF density of the lens) given MTF_lens, frequency f, and rolloff order lord.
    
    Equation: MTF_lens(f) = 1 / (1 + |f / flens|^lord)
    """
    if MTF_lens <= 0 or MTF_lens >= 1:
        raise ValueError("MTF_lens must be between 0 and 1 (exclusive).")
    
    flens = f / ((1 / MTF_lens - 1) ** (1 / lord))
    return flens

def mtf_curve(f50, fboost=0, flens=0, lord=2, dscan=0, sincpwr=0, ksharp=0):
    """
    Simulate MTF (Modulation Transfer Function) curve for film, lenses, scanners, and sharpening.
    """
    f = np.linspace(0.1, 200, 1000)
    
    mtf_film = 1 / (1 + (f / f50) ** 2)
    if fboost > 0 and fboost < 0.45 * f50:
        k1 = 1
        f50a = np.sqrt(f50**2 - 2*f50*fboost - fboost**2)
        mtf_film = k1 / (1 + ((f - fboost) / f50a) ** 2)
    
    if flens > 0:
        mtf_lens = 1 / (1 + (f / flens) ** lord)
    else:
        mtf_lens = np.ones_like(f)
    
    if dscan > 0:
        mtf_scanner = np.abs(sinc(f / dscan)) ** sincpwr
    else:
        mtf_scanner = np.ones_like(f)
    
    if ksharp != 0 and dscan > 0:
        rsharp = 1 if ksharp > 0 else 2
        mtf_sharp = (1 + np.abs(ksharp) * np.cos(2 * np.pi * f * rsharp / dscan)) / (1 + np.abs(ksharp))
    else:
        mtf_sharp = np.ones_like(f)
    
    mtf_total = mtf_film * mtf_lens * mtf_scanner * mtf_sharp
    
    # Compute f50 values dynamically for legend
    legend_labels = []
    if f50 > 0:
        legend_labels.append(f"Film MTF (f50 = {find_f50(f, mtf_film):.1f} lp/mm)")
    if flens > 0:
        legend_labels.append(f"Lens MTF (f50 = {find_f50(f, mtf_lens):.1f} lp/mm)")
    if dscan > 0:
        legend_labels.append(f"Scanner MTF (f50 = {find_f50(f, mtf_scanner):.1f} lp/mm)")
    if ksharp != 0:
        legend_labels.append(f"Sharpening MTF (f50 = {find_f50(f, mtf_sharp):.1f} lp/mm)")
    legend_labels.append(f"Total MTF (f50 = {find_f50(f, mtf_total):.1f} lp/mm)")
    
    # Print f50 values before plotting
    for label in legend_labels:
        print(label)
    
    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.semilogx(f, mtf_film, label=legend_labels[0], linestyle='dashed')
    if flens > 0:
        plt.semilogx(f, mtf_lens, label=legend_labels[1], linestyle='dotted')
    if dscan > 0:
        plt.semilogx(f, mtf_scanner, label=legend_labels[2], linestyle='dashdot')
    if ksharp != 0:
        plt.semilogx(f, mtf_sharp, label=legend_labels[3], linestyle='solid')
    plt.semilogx(f, mtf_total, label=legend_labels[-1], linewidth=2)
    
    plt.xlabel("Spatial Frequency (lp/mm)")
    plt.ylabel("MTF")
    plt.title("MTF Curve Simulation")
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()
    
    return f, mtf_total

# Example usage
if __name__ == "__main__":
    
    MTF_lens_example = 0.66
    f_example = 20  # line pairs/mm
    lord_example = 2
    estimated_flens = estimate_flens(MTF_lens_example, f_example, lord_example)
    print(f"Estimated flens for MTF = {MTF_lens_example}, f = {f_example}, lord = {lord_example}: {estimated_flens:.2f} lp/mm")


    f, mtf = mtf_curve(f50=45, fboost=0, flens=28, lord=2, dscan=4000/25.4, sincpwr=3, ksharp=0)

    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="MTF Curve Simulation")
    
#     parser.add_argument("--f50", type=float, required=True, help="50% MTF density of film (lp/mm)")
#     parser.add_argument("--fboost", type=float, default=0, help="Frequency shift of the 2nd order equalizer")
#     parser.add_argument("--flens", type=float, default=0, help="50% MTF density of the lens")
#     parser.add_argument("--lord", type=int, default=2, help="Lens rolloff order (default = 2)")
#     parser.add_argument("--dscan", type=float, default=0, help="Scanner pixels per mm")
#     parser.add_argument("--sincpwr", type=float, default=0, help="Power of sinc function (scanner MTF)")
#     parser.add_argument("--ksharp", type=float, default=0, help="Sharpening boost")

#     args = parser.parse_args()
    
#     mtf_curve(args.f50, args.fboost, args.flens, args.lord, args.dscan, args.sincpwr, args.ksharp)
