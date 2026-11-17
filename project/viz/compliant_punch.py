"""Render recorded contact states without advancing or altering the simulation."""
from __future__ import annotations
import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image


def plane_segments(positions, faces):
    """Exact linear-triangle intersections with world y=0, projected to x/z."""
    tri=positions[faces]
    a=tri[:,[0,1,2]];b=tri[:,[1,2,0]]
    denominator=b[:,:,1]-a[:,:,1]
    hit=(a[:,:,1]*b[:,:,1]<=0)&(np.abs(denominator)>1e-15)
    segments=[]
    for row in np.flatnonzero(hit.sum(axis=1)>=2):
        points=a[row,hit[row]]+(-a[row,hit[row],1]/denominator[row,hit[row]])[:,None]*(b[row,hit[row]]-a[row,hit[row]])
        points=np.unique(points,axis=0)
        if len(points)>=2:segments.append(points[[0,-1]][:,[0,2]])
    return segments


def render_case(case, output=None):
    case = Path(case)
    output = Path(output) if output else case/'render'
    output.mkdir(parents=True, exist_ok=False)
    meta=json.loads((case/'metadata.json').read_text())
    summary=json.loads((case/'summary.json').read_text())
    metrics=[json.loads(line) for line in (case/'metrics.jsonl').read_text().splitlines()]
    frames=[]
    for path in sorted(case.glob('frame_*.npz')):
        with np.load(path,allow_pickle=False) as d: frames.append({k:d[k] for k in d.files})
    if not frames: raise ValueError('no recorded frames')
    if any(not np.isfinite(f['positions']).all() for f in frames): raise ValueError('nonfinite frame')
    config=meta['config']
    local=np.array([float(f['time']) for f in frames])
    slowdown=10.
    images=[]
    fig=plt.figure(figsize=(12,7),facecolor='#f5f7fa')
    grid=fig.add_gridspec(2,2,width_ratios=[1.13,1],height_ratios=[1.35,1],
        left=.045,right=.96,top=.83,bottom=.12,wspace=.24,hspace=.34)
    ax3=fig.add_subplot(grid[:,0],projection='3d')
    cut=fig.add_subplot(grid[0,1]); curve=fig.add_subplot(grid[1,1])
    x0=frames[0]['positions']
    surface_mask=np.all(np.max(np.abs(x0[frames[0]['faces'],:2]),axis=1)<.145,axis=1)
    sphere_theta=np.linspace(0,2*np.pi,45); sphere_phi=np.linspace(0,np.pi,25)
    times=np.array([r['time'] for r in metrics]); ratios=np.array([r['penetration_ratio'] for r in metrics])
    for frame in frames:
        for ax in (ax3,cut,curve): ax.clear()
        pos,faces,center=frame['positions'],frame['faces'],frame['sphere_center']
        radius=float(frame['radius']); t=float(frame['time'])
        status=summary['status'] if int(frame['step'])==summary['completed_steps'] else 'recorded state'
        fig.suptitle('Sphere / cloth contact — visual experiment',fontsize=19,x=.5,y=.97,color='#152d46',weight='bold')
        # Keep the single changing subtitle outside axes.
        if len(fig.texts)>1:
            for old in list(fig.texts[1:]): old.remove()
        fig.text(.5,.89,f"{summary['preset']}  |  K={config['stiffness']:g}, c={config['contact_damping']:g}, I={config['iterations']}  |  "
            f"absolute t={float(frame['absolute_time']):.5f} s  (+{t:.5f} s)",ha='center',fontsize=11,color='#334b60')
        cloth=Poly3DCollection(pos[faces[surface_mask]],facecolor='#218b9c',edgecolor='none',alpha=.94)
        ax3.add_collection3d(cloth)
        sx=center[0]+radius*np.outer(np.cos(sphere_theta),np.sin(sphere_phi))
        sy=center[1]+radius*np.outer(np.sin(sphere_theta),np.sin(sphere_phi))
        sz=center[2]+radius*np.outer(np.ones_like(sphere_theta),np.cos(sphere_phi))
        ax3.plot_wireframe(sx,sy,sz,rstride=3,cstride=2,color='#d5813a',linewidth=.55,alpha=.55)
        ax3.set(xlim=(-.15,.15),ylim=(-.15,.15),zlim=(-.027,.245),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]')
        ax3.set_box_aspect((1,1,.9)); ax3.view_init(elev=18,azim=-61)
        ax3.set_title('Fixed camera · central cloth patch',fontsize=11,pad=15)
        # Linear triangle / sphere intersections with the same world y=0 plane.
        extent=np.sqrt(max(0.,radius**2-center[1]**2))
        xx=np.linspace(-.07,.07,400)
        bottom=center[2]-np.sqrt(np.maximum(0.,extent**2-(xx-center[0])**2))
        cut.fill_between(xx,bottom,.03,color='#eda86b',alpha=.28)
        cut.plot(xx,bottom,color='#c16e29',lw=2,label='Sphere surface')
        cut.add_collection(LineCollection(plane_segments(pos,faces),colors='#08798c',linewidths=2,label='Cloth surface'))
        cut.add_collection(LineCollection(plane_segments(x0,faces),colors='#8395a7',linewidths=1,linestyles=':',label='Parent state'))
        cut.set(xlim=(-.07,.07),ylim=(-.021,.012),xlabel='x [m]',ylabel='z [m]')
        cut.set_title('Contact close-up · exact section at y=0',fontsize=11)
        cut.grid(alpha=.18); cut.legend(loc='lower left',fontsize=8,frameon=False)
        if len(times): curve.plot(times,100*ratios,color='#8fa7b2',lw=1,alpha=.65)
        mask=times<=t+1e-12
        curve.plot(times[mask],100*ratios[mask],color='#117f91',lw=2)
        curve.axhline(2,color='#bc4559',ls='--',lw=1,label='2% stop guard')
        curve.axvline(t,color='#d5813a',lw=1)
        curve.set(xlim=(0,max(.025,float(local[-1]))),ylim=(0,max(2.2,100*ratios.max(initial=0)*1.1)),
            xlabel='Branch time [s]',ylabel='Penetration / R [%]')
        curve.set_title(f"Whole-triangle depth: {float(frame['penetration_ratio'])*radius*1000:.3f} mm",fontsize=11)
        curve.grid(alpha=.18); curve.legend(loc='lower right',fontsize=8,frameon=False)
        fig.text(.045,.038,f"{status}  |  playback {slowdown:g}x slower  |  visual use; not engineering validated",
                 fontsize=10,color='#773c36' if 'stopped' in status else '#43586b')
        buffer=BytesIO(); fig.savefig(buffer,format='png',dpi=110,facecolor=fig.get_facecolor()); buffer.seek(0)
        im=Image.open(buffer).convert('RGB'); images.append(im.copy()); buffer.close()
    images[-1].save(output/'contact.png')
    # GIF stores centiseconds. Quantize cumulative time, not each interval,
    # so 15 samples of 1/120 s still play for 1.25 s at 10x slowdown.
    ticks=np.rint((local-local[0])*slowdown*100).astype(int)
    durations=[max(10,int(d)*10) for d in np.diff(ticks)]+[800]
    images[0].save(output/'contact.gif',save_all=True,append_images=images[1:],duration=durations,loop=0)
    plt.close(fig)
    (output/'render.json').write_text(json.dumps(dict(source=str(case.resolve()),frames=len(frames),
        physical_duration=float(local[-1]-local[0]),slowdown=slowdown,final_hold_ms=800,
        encoded_playback_ms=sum(durations[:-1]),gif_duration_ms=durations,
        scientific_render=True,geometry_interpolation=False,engineering_validated=False),indent=2),encoding='utf-8')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('case',type=Path); p.add_argument('--output',type=Path)
    a=p.parse_args(); print(render_case(a.case,a.output))


if __name__=='__main__': main()
