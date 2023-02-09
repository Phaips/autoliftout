# User Guide

TODO: add images
AutoLiftout is an automated liftout program for the preparation of cryo-lamella.

## The Workflow

The AutoLiftout workflow consists of a number of stages that must be completed sequentially. 

Setup:
- The user selects lamella and landing positions.
- Optionally sputter platinum to protect the sample.

MillTrench:

Mill lower, upper and side trenches using high currents. (horsheshoe pattern, see ref)
- Lower and Upper trenches release lamella from the bulk.
- Side trench provides access for the needle.

Mill Undercut: 

- Mill the underside and part of the other side of the lamella to release from the base of the bulk.

Liftout: 

1. Insert the needle.
2. Guide the needle to near the lamella. 
3. Charge the sample with the ion beam
4. Make contact with the lamella, 
5. Sever the lamella from the bulk.
6. Retract the needle 

Landing:
1. Insert the needle
2. Guide the lamella to the post
3. Weld lamella to the post
4. Discharge the lamella with the electron beam
5. Retract the needle

Polishing Setup:
- The user selects the regions of the lamella to polish

Thinning:
- Thin the lamella to a lower thickness, using a relatively high current.

Polishing:
- Polish the lamella to electron transparent thickness.
- Should be completed immediately prior to transfer to prevent contamination buildup. 

## Methods
We have developed a number of different methods for performing liftout. We believe these methods provide a more reliable, and higher throught liftout workflow. The methods developed work are complementary.

For more background on these methods, please read: AutoLiftout: A manufacturing approach.

**Manipulator Preparation**

We flatten the side of the needle to prepare for the maximum surface contact area with the side of the lamella.

**Landing Surface Preparation**

We flatten the side of the landing post to prepare for the maximum surface contact area with the side of the lamella.


**Charge Control**

Biological samples in cryogenic conditions often have large amounts of charge which causes challenges with imaging. Often this charging saturates the detectors and images either glow white or completely dark. To overcome this issue, we have developed the following techniques:

- AutoGamma: We automatically apply a digital gamma correction to the image if sufficient mean pixel intensity is determined. Gamma correction shifts the image histogram allowing features to be detected in the image (however it reduces image quality).
- Charge Neutralisation: To reduce the accumulated charge, we apply a charge neutralisation procedure (e.g. taking a rapid series of electron images to neutralise ion charge after large milling operations). This helps control the charge buildup throughout the process. 

**Big Lamella**

To provide better landing stability, we liftout much larger lamella than typical. In conjunction with Manipulator Preparation, Landing Surface Preparation, and Side Pickup we are able to make consistent, right-angled contact with a large surface area between the lamella and the landing post. This provides a more stable base from which to thin the lamella down. The downside of using this method is an increased material waste, and increased thinning time to remove excess material.

**Side Pickup**

To provide better liftout, and landing stability we make contact with the side of the lamella to lift it out of the trench. The side pickup provides the following benefits:
- More stability on contact: we apply a compressive force to the side of the lamella, instead of bending (if touching from the top). This allows us to make firmer contact without bottoming out the lamella in the trench. 
- Better orientation for landing: Due to the lack of rotation/tilt control the angle of liftout determines the orientation for landing the lamella. When lifting from the top of the lamella, it can sometimes rotate, roll or slide of the needle tip when making contact with the post causing bad landing orientation. It is analogous to spinning a basketball on your finger, it can be done but is difficult to repeat. Making contact from the side allows for the lamella to be evenly compressed between the needle and the post, maintaining its orientation.

**Contact Detection** 
In order to determine whether the needle and lamella have made sufficient contact, we developed a contact detection procedure. 
- We monitor the image brightness, whilst driving the needle towards the lamella.
- When contact is made between the needle and lamella, there is a significant increase in brightness due to charge disipation. This effect occurs due to the charge build up in the platinum crust being grounded when contact is made with the needle.
- We detect this change, and stop the needle movement. 

**Charge Pickup**
We have developed a repeatable procedure for lifting the lamella by only manipulating the charge buildup. This method does not rely on platinum deposition, or welding (redeposition). 
- To attach: We move the needle and lamella close together, and take a series ion beam images to build up charge. When the lamella and needle make contact they stick together with static due to charge. 
- To dettach: Once the lamella is welded to the landing post, we run the electron beam to disapate the charge, and the needle slides off the lamella. 

The procedure is still being developed, and understood and is very sensitive to parameters and different conditions (e.g. the number of images being taken).


### Automation
The program uses the following features to progress through the workflow.
 
**State Restoration**

State restoration provides a way to ‘checkpoint’ the position of the workflow. The state is saved at the end of each stage, and can be restored at any time. 
- When continuing to the next stage, the program will initially restore to the previous microscope state before continuing.
- This allows the user to restart, pause or close the program and continue from where they left off (assuming no catastrophic lamella damage). 
- This also allows multiple lamella stages to be batch produced, e.g. perform all trench milling in a row. Note: Some lamella stages must be completed sequentially (e.g. Liftout -> Landing). 

**Alignment**

The program uses a fourier cross-correlation to align to reference images. 
- This alignment is used when a quality reference image is available, for example alignment after restoring state to account for hardware limitations.
- In general the alignment will use the stage movement to correct, but a higher precision beam shift alignment will be used for higher precision tasks.

**Feature Detection**

The program uses a segmentation model to detect common features in the workflow, these include the Needle Tip, the Lamella and the Landing Post.
- These feature positions are used to guide the program decision making. For example, the model is used to guide the needle tip to make contact with the lamella. 
- When supervising the user can correct these feature detections using an integrated user interface. This is discussed in the next section.

### User Interface

We provide a user interface to enable the user to run the microscope and autoliftout with minimal training.

**Ease of Use**

The launch ui is where you will start when you open autoliftout. It allows you to create, and load experiments and protocols, as well as use stand alone tools such as sputtering platinum and settings validation.

Movement
The movement ui allows the user to double click on the image to move the stage to the desired location. Two movement modes are available; Stable movements will maintain the eucentricty (ensure both beams are focused at the same point), while Eucentric movements will constrain the stage to move vertical to move the stage back to the eucentric point. When moving eucentricly, the user should first centre a feature in the Electron view, and then double click the same feature in the Ion view to correct the eucentricity. 

Milling
The milling ui provides control over the pre-defined patterns used in autoliftout. The user can adjust all the parameters of these patterns to suit their needs. These parameters can be saved on the go to the protocol file.

**Supervision Mode**
Users can choose to supervise the workflow by enabling it in the protocol. In this mode, the program will automatically perform all the movements and operations, but will pause and ask the user for confirmation before continuing. Supervision can be turned on/off for individual stages. 

Feature Detection
An example of supervision is the feature detection interface. While in supervision mode, the program will show the user the detected features, and the movement plan. The user can simply click to correct the detected feature, and change the calculated movements. 

Other supervised steps will ask the user to confirm milling operations, and enable the movement interface to correct alignments. 


### Adapting to New Use Cases
AutoLiftout has been designed to be adapted to a range of samples, conditions, and systems. We have attempted to provide a number of variables

**Changing the System**

The system specification and settings can be changed by editing the system.yaml file. 
- This file defines the system configuration, and the beam settings you want to use for the run. 
- For example, you may choose to use a different detector mode, or type or a different plasma gas. 
- In particular, the stage definitition is important as it defines the relative coordinates of the stage and beams. If you're microscope is different you will need to adjust these parameters. 

**Changing the Protocol** 

The user can manually edit the protocol.yaml file to change any of the available parameters.
- When using the user interface, changes to the milling parameters will prompt the user to update their protocol file. 
- New protocols can be loaded in the launch user interface. 

**Model Retraining**
- The provided model can be finetuned for a new dataset (sample, conditions, etc). 
- The most efficient way to collect a new dataset is to run autoliftout in supervised mode using your desired sample/conditions. Whenever the user corrects a feature detection using the interface, the program will automatically flag that image for labelling and save it in a separate directory. 
- The user can then use OpenFIBSEM to label (labelling tool) and finetune the model on the new dataset. For more information on these tools please see the OpenFIBSEM repository.

**Automation Tuning**
- Variables used for automation can be changed by editing the code.
- Depending on your application you might need to adjust the cross-correlation masks and filter strengths, or step-sizes, or number of iterations.
- In a future release these will be separated into an external configuration file for easier editing.