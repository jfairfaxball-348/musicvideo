"""Frame-exact conform, final soundtrack mux, decoded media QA and contact sheets."""
from fractions import Fraction
import json, subprocess
import av
import numpy as np
from PIL import Image, ImageDraw
from render import CLIPS, LOCAL, OUT, FF, FPS, FINAL, inspect, run, write, source, PLAN

def pcm_hash(path):
    return subprocess.check_output([FF,"-v","error","-i",str(path),"-map","0:a:0","-c:a","pcm_s24le","-f","hash","-"]).decode().strip()

def conform(shots,t,dest):
    clock=3528000
    ticks=clock//FPS
    end=round(t["samples"]*clock/t["sample_rate"])
    with av.open(str(dest),"w",options={"video_track_timescale":str(clock),"movie_timescale":str(clock)}) as out:
        stream=None
        for s in shots:
            path=CLIPS/f"{s['id']:04d}.mp4"
            with av.open(str(path)) as clip:
                vs=clip.streams.video[0]
                if stream is None:
                    stream=out.add_stream_from_template(vs)
                    stream.time_base=Fraction(1,clock)
                i=s["start_frame"]
                for p in clip.demux(vs):
                    if p.dts is None: continue
                    p.stream=stream
                    p.time_base=Fraction(1,clock)
                    p.pts=p.dts=i*ticks
                    p.duration=min(ticks,end-i*ticks)
                    out.mux(p); i+=1
                if i!=s["end_frame"]: raise ValueError(f"Frame count at shot {s['id']}")

def validate(path,shots,t):
    with av.open(str(path)) as c:
        v=c.streams.video[0]; a=c.streams.audio[0]
        duration=Fraction(t["samples"],t["sample_rate"])
        assert v.codec_context.name=="h264" and (v.width,v.height)==(1920,1080)
        assert abs(v.duration*v.time_base-duration)<=v.time_base
        assert abs(a.duration*a.time_base-duration)<=a.time_base
        targets={(s["start_frame"]+s["end_frame"])//2:s for s in shots}
        frames=0; sheets=[]; thumbs=[]; dark=0
        for f in c.decode(v):
            if abs(f.pts*f.time_base-Fraction(frames,FPS))>v.time_base: raise ValueError("Presentation timestamp discontinuity")
            gray=f.reformat(width=96,height=54,format="gray").to_ndarray()
            if gray.std()<4 or gray.mean()<5: dark+=1
            if frames in targets:
                s=targets[frames]
                im=f.reformat(width=240,height=135,format="rgb24").to_image()
                if any(np.abs(np.asarray(im,dtype=float)-k).mean()<1 for k in thumbs): raise ValueError("Duplicate shot thumbnail")
                thumbs.append(np.asarray(im,dtype=float))
                if (len(thumbs)-1)%24==0: sheets.append(Image.new("RGB",(960,6*159)))
                idx=(len(thumbs)-1)%24; x,y=idx%4*240,idx//4*159
                sheets[-1].paste(im,(x,y))
                ImageDraw.Draw(sheets[-1]).text((x+3,y+138),f"{s['id']:03d}  {s['start']:.1f}s  {s['subject']}",fill="white")
            frames+=1
        assert frames==t["frames"] and dark==0
        for i,sheet in enumerate(sheets): sheet.save(OUT/f"contact-{i+1:02d}.jpg",quality=90)
    with av.open(str(path)) as c: samples=sum(f.samples for f in c.decode(audio=0))
    assert samples==t["samples"]
    audio=pcm_hash(path)
    assert audio==pcm_hash(source())
    return dict(frames=frames,samples=samples,duration=float(duration),audio_pcm24=audio,black_frames=dark,shots=len(shots),dimensions=[1920,1080],codec="h264",soundtrack="ALAC, original decoded 24-bit PCM",contact_sheets=len(sheets))

def assemble(shots,t):
    picture=LOCAL/"conform.mp4"
    conform(shots,t,picture)
    pending=OUT/"review.mp4"
    run([FF,"-v","error","-y","-i",picture,"-i",source(),"-map","0:v:0","-map","1:a:0","-map_metadata","-1","-c:v","copy","-af","aformat=sample_fmts=s32","-c:a","alac","-video_track_timescale","3528000","-movie_timescale","3528000","-movflags","+faststart",pending],"assembly.log")
    result=validate(pending,shots,t)
    pending.replace(FINAL); write(OUT/"validation.json",result)
if __name__=="__main__":
    assemble(json.loads((PLAN/"shots.json").read_text()),json.loads((PLAN/"timeline.json").read_text()))
