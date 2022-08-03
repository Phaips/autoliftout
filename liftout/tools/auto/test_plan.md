### Testing Plan

TODO: START_HERE: add more detail and example scripts to run
autoliftout test plan
for after maintenance

MoveSettings
- tilt_compucentric
- link_z_y

new stage movement
- test the new move_stage_relative_with_correct_movement function using the movement_window
- test that it continues to move perfectly.

new needle movment
- insert needle
- move to liftout / landing position
- open detection window
- test movement with eb and ib
- try to move the needle to the centre in each view

new eucentric movement


new crosscorrelation
- test the new cross correlation function
- can probably do this offline
- make a better tool for this

auto-eucentric workflow

new autolifout flow



CHECK MOVE SETTINGS!!!! NEED TO TEST if this fixes problems
should hopefully fix tilt alignments??


STEP PLAN
event map

Setup
- manual


Trench
- move to lamella coordinates         
- tilt to trench angle
- correct position * (align / take images)
- mill trench                               # CRITICAL
- take reference images (ref_trench*)

JCut
- move to lamella coordinates
- take reference images 
- tilt flat to electron
- correct position (align / take images)
- tilt to jcut angle 
- correct position (algin / take images)
- mill jcut                                 # CRITICAL
- take reference images (ref_jcut*)
- tilt flat to electron
- take reference imags (ref_jcut*)

Liftout
- move to lamella coordinate
<!-- - tilt to liftout angle -->
- correct position
- insert needle
- land_needle_on_milled_lamella
-   align needle eb (xy)
-   move in z (halfway... can probs manually set this)
-   loop:
-       while brightness not above history limit
-       continue moving down towards sample        # CRITICAL
-    take reference images
- sputter platinum
- mill jcut sever                                  # CRITICAL                           
- remove needle from trench
- retract needle
- take reference images (ref_liftout_lamella*)

Landing
- move to landing coordinate
- link stage ?
- eucentric correction
- insert needle
- align needle eb (xy)
- align needle ib (z)
- align needle ib half (x)
- loop:
    - align needle ib (x) until landed        # CRITICAL
- mill weld
- mill cut needle                             # CRITICAL
- take reference images
- remove needle
- retract needle
- take reference images (ref_landing_lamella*)

Reset
- move stage out
- insert needle
- centre needle
- mill sharpen needle
- retract needle
- move stage back in

Thin
- move to landing coordinates
- eucentric correction
- move to thinning angle (tilt, rotation)
- eucentric correction
- align to ref (rotated), ib 
- mill thin lamella                              # CRITICAL
- take reference images (ref_thin_lamella)


Polish
- move to thinning coordinates
- align to ref (ref_thin_lamella)
- mill polish lamella                            # CRITICAL
- take reference images
