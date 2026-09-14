## Experimental protocol

We have 3 different use cases for the concealer.

UC1: The concealer is ran directly on headphones, without any connexion to a 
computer (same as noise canceling on headphones). It can be coupled with
noise cancelling. It needs to run with low computationnality. 
We can try using:
vad: webrtc-vad
mic: headphones mic
concealer: simple reverse (with webrtc-vad for selecting segments)
We can emulate that, but we will nether be able to run the code
on the headphones hardware without the constructor help.

UC2: The concealer is used with headphones, but ran on computer. Then we can try
using:
vad: ten vad, silero vad
mic: headphones mic (close), computer mic (distant)
concealer: simple reverse method (with ten vad or silero vad for selecting segments), tts auto method

UC3: The concealer is used on a specific device with directional speakers, 
with integrated microphone. We very very likely need ultrasonic directional 
speakers, but they are very pricy. We can try DIY, though https://www.youtube.com/watch?v=B8ss4KqcuXU&t=1s
The problem is that I don't see any point of UC3 for our applications, it seems so complicated 
compared to just putting headphones on.

Same as UC2, we can try using:
vad: ten vad, silero vad
mic: headphones mic (close), computer mic (distant)
concealer: simple reverse method, tts auto method

To evaluate the different models, we need to:
1) create realistic office audio environments, with background unintelligible voice
2) add a salient intelligible voice that is talking for example as in a zoom meeting.
We might want to use a tts specialized in natural sounding voices.
3) create a head with polystyrene with a microphone in each ear, and run the
concealer in stream mode (either with headphones for UC1 and UC2, or with 
speakers for UC3)
4) calibrate the speakers with a sound level meter
5) play the designed sounds on stereo speakers in a controlled environment 

At first, the audios need to be recorded blankly, without any concealing.
Then, the other recordings depend on the UC's.

### For UC1 and UC2

We need to make 4 different recordings: without any headphones, with headphones, with 
headphones and noise canceling on, with headphones and noise canceling on
and speech concealing on.

After that we can use the signal on both ears to make objective 
and subjective evaluation of the algorithm. 

### For open concealing (with speakers)

We need the device already built. We need to make 2 different recordings: 
without anything, and with concealing.

## Ideas

One idea of something we can try is that the tts model (pocket-tts) is still very heavy (100M parameters) a will probably not work on 
a smartphone for example. But this TTS is also overkill, as we don't really care about the words that are spoken, we just want to 
capture the prosody and the timbre, and then the words can be nonsensical. Which means that the part where the text token are 
interpreted can be removed. Or maybe we could try to develop a model that takes a voice as input and in real-time 
retransforms it into a simlish voice.