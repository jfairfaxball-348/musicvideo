import json, random
from analyse import PLAN, FPS, write

CAST = {
 "vocalist":"Wiry olive-skinned frontman, dense halo of black corkscrew curls, burgundy sleeveless tank exposing bare shoulders; black handheld wired microphone with a small round silver grille. At downstage centre.",
 "guitarist":"Black woman with a close buzz cut, faded mustard work shirt open over a black tank, battered cream solid-body electric guitar with black pickguard. At audience left.",
 "bassist":"Broad-shouldered pale man with long copper hair, navy mechanic coveralls, black four-string bass on a canvas strap. At audience right.",
 "drummer":"Muscular brown-skinned man with shaved head, white sleeveless tee, large oxblood acoustic drum kit and brass cymbals on the centre rear riser.",
 "stage":"Enormous open-air festival in a dusty field, scaffold towers, black stage drape, four musicians on stage, amber PAR cans, audience stretching to the tree line.",
 "crowd":"Dense sea of young adult festival fans in faded cotton shirts behind a steel crush barrier, long-haired fans, fists in air, late afternoon dust over distant trees."
}
LOOK="Candid late-1990s live concert documentary footage. Breezy outdoor festival, warm amber sidelight, weathered black stage drape, sweaty skin, deep shadows, dusty 35mm film, natural motion blur, photorealistic."
SHOTS={
 "vocalist":[("side profile from pit, 85mm","Curls bounce as the singer lunges over the stage lip and spits rhythmic lines into the mic hiding his mouth."),("low shoulder camera","The frontman paces across the stage and punches a fist overhead, cable trailing behind."),("over his shoulder facing the field","The singer thrusts the mic out toward an immense answering crowd."),("tight backlit silhouette","His shoulders shake with each accented line as he grips the microphone.")],
 "guitarist":[("low angle at the stage lip","She digs into a syncopated riff, shoulders rocking with each downstroke."),("tight handheld insert across the neck","The picking hand chops across the strings of the chipped cream electric guitar."),("three-quarter profile, waist up","She steps into the light as she bends a note, neck of the guitar rising.")],
 "bassist":[("medium shot looking across stage","The bassist leans back into the groove, copper hair swinging as he plucks."),("close from below the headstock","The black bass rocks across his hip while his fingers work the strings."),("waist-up side tracking shot","He walks toward the drum riser with knees bent in time.")],
 "drummer":[("beside the floor tom, 28mm","The drummer alternates snare and tom hits; cymbal edges tremble in the amber light."),("tight over the drummer's shoulder","Sticks strike the skins in a rapid fill then crash into a brass cymbal."),("low side profile framed through stands","The drummer leans over the oxblood kit, elbows rising and falling.")],
 "crowd":[("handheld eye level inside the crowd","People surge and jump together, dust flying as fists punch the air."),("telephoto across a sea of heads","Thousands of raised fists ripple from the barrier toward the distant trees."),("elevated oblique view","A circular mosh pit opens in the crowd; dancers run around the dusty ring."),("low camera at the front barrier","Excited sweaty fans shout the refrain and reach toward the lens."),("moving shoulder camera behind the front row","A crowd surfer floats face-up on raised hands past the camera."),("long lens from a scaffold","Waves of jumping spectators break across the field on the downbeat.")],
 "stage":[("very wide elevated crane view from stage rear","A four-piece band faces an immense crowd stretching to the horizon; the camera sweeps slowly across the stage."),("wide front-of-house camera","The four musicians play in amber haze under a tall scaffold as audience silhouettes bounce in the foreground."),("wide lateral view from the stage wing","A cable-strewn wooden stage vibrates as the band plays, beyond them an ocean of moving heads.")]
}
def main():
    t=json.loads((PLAN/"timeline.json").read_text())
    write(PLAN/"identity.json",dict(cast=CAST,look=LOOK,base_seeds={k:78123+j*119 for j,k in enumerate(CAST)}))
    rng=random.Random(199807)
    beat=[round(b*FPS) for b in t["beats"]]
    cuts=[0]
    subjects=[]
    count={k:0 for k in CAST}
    shots=[]
    for seg in t["sections"]:
        end=round(seg["end"]*FPS) if seg!=t["sections"][-1] else t["frames"]
        while cuts[-1]<end:
            a=cuts[-1]
            info=t["half_second_features"][min(len(t["half_second_features"])-1,int((a/FPS+1)*2))]
            step=rng.choice([4,6,8]) if info["energy"]>.7 else rng.choice([6,8,8])
            upcoming=[b for b in beat if b>a+2]
            b=min(end,upcoming[step-1] if len(upcoming)>=step else end)
            if end-b < FPS*1.5: b=end
            choices=["guitarist","bassist","drummer"] if info["vocal_ratio"]<.23 else ["vocalist","vocalist","crowd","guitarist"]
            if seg["section"]=="chorus candidate" or info["energy"]>.80: choices+=["crowd","stage","drummer"]
            sub="stage" if a==0 else rng.choice([s for s in choices if s!=subjects[-1]])
            if b==t["frames"]: sub="stage"
            j=count[sub]; count[sub]+=1
            angle,action=SHOTS[sub][j%len(SHOTS[sub])]
            light=rng.choice(["A shaft of amber light cuts through the dust.","Sunlight rims the silhouettes.","A wash of dull red crosses the shadows."])
            prompt=f"{angle.capitalize()}. {action} {CAST[sub]} {light} {LOOK}"
            shots.append(dict(id=len(shots)+1,start_frame=a,end_frame=b,start=a/FPS,end=min(b/FPS,t["duration"]),subject=sub,action=action,camera=angle,prompt=prompt,section=seg["section"],energy=info["energy"],vocal_ratio=info["vocal_ratio"],technique="Wan VACE reference-to-video",seed=rng.randrange(1,2**30)))
            cuts.append(b); subjects.append(sub)
    write(PLAN/"shots.json",shots)
if __name__=="__main__": main()

