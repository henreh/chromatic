import copy, random, math
from music21 import stream, chord, note, scale, pitch, harmony

"""
Collection of utility functions and classes supporting chromatic functionality. 
"""

class ChordProgression(object):
	"""
	A chord progression represents the idea of a harmonic framework -- i.e. a set of notes and chords scaffolding the rest of the piece
	at any one time, even if they're not played themselves as part of the piece. 

	The ChordProgression object provides functions for determining what notes are present in the chord progression at any given time offset,
	for use in determining what notes might be consonant or not consonant, for example. 

	A ChordProgression can be created from two objects: (i) a music21 stream, or (ii) a music21 scale. If (ii), the scale is converted internally
	to a stream with the notes of the scale played all at once. If (i), the stream will be internally looped -- i.e. if a 4-bar stream is given, and
	the user asks what notes are playing in the chord progression at offset 33.0, this will be equivalent to offset 1.0. 

	If the stream given to a ChordProgression contains sections in which no notes are playing, behaviour is undefined and may cause exceptions in 
	many parts of chromatic. 
	"""

    def __init__(self, chordProgression):
    	"""
    	Args:
    		chordProgression: Music21 stream or scale object to build the chord progression
    	"""
    	cpType = "STREAM" if type(chordProgression) is stream else "SCALE"
        self.type = cpType
        if cpType == "STREAM":
            self.root = ""
            self.cp = chordProgression
        
        if cpType == "SCALE":
            #Build a stream out of the scale
            pitches = chordProgression.getPitches('c1', 'c12')
            newStream = stream.Stream()
            ch = chord.Chord(pitches)
            ch.duration.quarterLength = 64.0
            newStream.insert(0, ch)
            self.cp = newStream
            self.root = chordProgression.pitchesFromScaleDegrees([0])
            
    def getRoot(self, n=None):
    	"""
    	Returns the root of the currently playing set of notes

    	Args:
    		n (note): optional note. If given, function returns root of chord playing when n sounds. If not given, function returns root of first chord.
    	"""
        if n is None:
            n = note.Note()
            n.offset = 0.0
        if self.type == "STREAM":
            return self.notesPlayingWhile(n)[0]
        else:
            return self.root
      
    def notesPlayingWhile(self, n, sort = True, doDuration = False):
    	"""
    	Returns a list of the pitches that are currently playing at the 
    	same time as a given element. If they chord progression is
    	shorter than the note allows then the chord progression
		loops.

		Args:
			n (note): Note to check concurrently playing chord for
			sort (boolean): If sort is true, result is sorted by pitch (default True)
   			doDuration (boolean): If doDuration is true (default), notes are checked for the entire duration of the checked note
    								if it is false, we just check what's playing at the starting onset of the event
		"""
        if self.cp.duration.quarterLength == 0.0:
            return None
        
        if type(n) == pitch.Pitch:
            n = note.Note(n)
            n.offset = 0
        
        #Deep copy doesn't preserve offset??
        oldOffset =  n.offset
        n = copy.deepcopy(n)
        n.offset = oldOffset
        
        if (n.offset >= self.cp.duration.quarterLength):
            n.offset = divmod(n.offset, self.cp.duration.quarterLength)[1]
         
        #All playing while sounding is a bit buggy at boundary changes, so roll our own   
        endOffset = n.offset + 0.01 if not doDuration else n.offset + n.duration.quarterLength
        pitches = self.cp.getElementsByOffset(n.offset,
                                              offsetEnd= endOffset,
                                              includeEndBoundary=False,
                                              mustFinishInSpan=False,
                                              mustBeginInSpan=False,
                                              includeElementsThatEndAtStart=False,
                                              classList=[note.Note, chord.Chord])
        toReturn = list(set(pitches)) #Remove duplicates
        if (sort):
            toReturn.sort(key=midiSort)
        return toReturn

    def notesPlayingWhileRange(self, n, minPitch, maxPitch, sort = True):
    	"""
    	Returns a list of the pitches that are currently playing at the 
    	same time as a given element within a given range of pitches. 
    	"""
        chordProgression = self.cp
        if (n.offset >= chordProgression.duration.quarterLength):
            n.offset = divmod(n.offset, chordProgression.duration.quarterLength)[1]
            
        pitches = allOctaves(chordProgression.allPlayingWhileSounding(n).pitches, 1, 6)
        toReturn = []
        
        for p in list(pitches):
            if p.midi >= minPitch.midi and p.midi < maxPitch.midi:
                toReturn.append(p)
                
        toReturn = list(set(toReturn)) #Remove duplicate
        if (sort):
            toReturn.sort(key=midiSort)
        return toReturn
    
    def consonantWhilePlaying(self, n, sort = True):
    	"""
    	Returns a list of pitches that are consonant with those that are currently
    	playing at the same time as a given element (including the chord pitches themselves)
		3 s/t = minor 3rd
		4 s/t = major 3rd
    	7 s/t = perfect 5th
    	8 s/t = minor 6th
		9 s/t = major 6th
		12 s/t = (perfect) octave

		This is quite a naive conception of consonance -- chromatic includes other more nuanced tools
		for determing consonance, if necessary. 
    	"""
        if self.type == "STREAM":
            consonantIntervals = [0,2,3,4,5,7,8,9,12]
            toReturn = []
            off = n.offset
            if off < 0:
                off = 0
            notesPlaying = self.notesPlayingAtOffset(off + 0.01, sort)
            baseOctave = notesPlaying[0].midi / 12
            for possibleNote in range(0,12):
                valid = True
                for testNote in notesPlaying:
                    #dist = abs(possibleNote - (testNote.midi%12))
                    dist = abs((testNote.midi%12) - possibleNote)
                    
                    if dist not in consonantIntervals:
                        valid = False
                        
                if valid:
                    p = pitch.Pitch()
                    p.midi = possibleNote + (12 * baseOctave)
                    toReturn.append(p)
                    
            toReturn = allOctaves(toReturn, baseOctave-1, baseOctave+1)
            return toReturn
        else:
            return allOctaves(self.notesPlayingAtOffset(n.offset + 0.01, sort), (n.midi/12)-1, (n.midi/12)+1)
    
    def getDegrees(self, n, includeSteps):
    	"""
		Given a list of chord degrees (includeSteps), returns a list of notes playing
    	at the same time as n that match those degrees
    	"""
        playingNotes = self.notesPlayingWhile(n, False)
        currChord = chord.Chord(playingNotes)
        cons = [currChord.getChordStep(i) for i in includeSteps]
        cons = [i if i is not None else n for i in cons]
        return cons
    
    def getChordScale(self, n):
    	"""
    	Returns the chord scale as a chord progression with degree 1 being the root
    	of the currently playing chord.

    	Todo:
    		* Explain idea of chord scale better
    	"""
        chordPlaying = chord.Chord(self.notesPlayingWhile(n))
        root = chordPlaying.root()
        chType = harmony.chordSymbolFigureFromChord(chordPlaying, True)
        n.show("text")
        if "major" in chType[1]:
            print root.name + "major"
            return ChordProgression(scale.MajorScale(root), 'SCALE')
        if "minor" in chType[1]:
            print root.name + "minor"
            return ChordProgression(scale.MinorScale(root), 'SCALE')
        
        s = stream.Stream()
        s.insert(0, chordPlaying)
        return ChordProgression(s)
    
    def getChordScaleDegreesAsChord(self, n, degrees):
    	"""
		Works out the current chord scale being played, and then returns the given scale
    	degrees as a chord (e.g. useful for a note pool)
    	"""
        chordScale = self.getChordScale(n)
        notes = []
        for i in degrees:
            notes.append(chordScale.degreesUp(n, i-1))
        return chord.Chord(notes)
    
    def fit(self, s):
    	"""
    	Ensures that all notes and chords in input stream s
    	are weakly consonant intervals with the progression. Pitches are moved
    	to closest consonant interval
    	"""
        for n in s.notes:
            cons = allOctaves(self.consonantWhilePlaying(n),1,6)
            if type(n) is note.Note:
                if n.midi%12 not in [no.midi%12 for no in cons]:
                    n.midi = self.nearestNote(n, cons)
            if type(n) is chord.Chord:
                for p in n.pitches:
                    if p.midi%12 not in [no.midi%12 for no in cons]:
                        p.midi = self.nearestNote(p, cons)
                        
        return s
    
    def setOctaves(self, lowOctave, highOctave):
    	"""
    	Re-works the chord progression to span more/less octaves, between
    	lowOctave and highOctave
    	"""
        newChordProg = stream.Stream()
        for n in self.cp:
            notes = allOctaves(n.pitches, lowOctave, highOctave)
            newChord = copy.deepcopy(n)
            newChord.pitches = notes
            newChordProg.insert(n.offset, newChord)
        
        self.cp = newChordProg
        return self
        
    def nearestNote(self, n, allowed):
    	"""
		Moves the pitch of n to the nearest pitch in allowed (list of note objects)
    	"""
        nOctave = n.octave
        nInvariant = n.midi
        
        random.shuffle(allowed) ##Shuffle allowed notes so we don't always pick first in ties
        
        currMin = 999999
        currValue = nInvariant
        for poss in allowed:
            if abs(nInvariant - poss.midi) < currMin:
                currValue = poss.midi
                currMin = abs(nInvariant - poss.midi)
        return currValue
      
    def degreesUp(self, n, stepSize):
    	"""
		Treats the notes currently playing as a scale. Firstly, get a list of notes playing
    	Then find the index of the nearest note to the given note. If the given note is a chord,
    	take the root.
		Then return that index + stepsize
		"""
        allowedUp = allOctaves(self.notesPlayingWhile(n), 1, 6)
        
        if type(n) is chord.Chord: n = n.root()
        
        currMin = 100
        currIndex = -1
        i = 0
        for testNote in allowedUp:
            if currMin > abs(testNote.midi - n.midi):
                currMin = abs(testNote.midi - n.midi);
                currIndex = i
            i += 1
        if currIndex == -1:
            return n
        
        returnIndex = currIndex + stepSize
        if returnIndex >= len(allowedUp) or returnIndex < 0:
            return n
        
        return allowedUp[returnIndex]
            
    def next(self, direction='ascending', stepSize=1, pitchOrigin=None):
    	"""
    	Get the next pitch above (or if direction is descending, below) a pitchOrigin or None.
    	If the pitchOrigin is None, the tonic pitch is returned. This is useful when starting a chain of iterative calls.
    
    	The direction argument may be either ascending or descending. Default is ascending.
    	Optionally, positive or negative integers may be provided as directional stepSize scalars.
      
    	An optional stepSize argument can be used to set the number of scale steps that are stepped through.
    	Thus, .next(stepSize=2) will give not the next pitch in the scale, but the next after this one.
    	"""
        chordProgression = self.cp
        
        if pitchOrigin is None:
            for n in chordProgression.notesAndRests:
                return n
        if type(pitchOrigin) is chord.Chord:
            pitchOrigin = note.Note(pitchOrigin.root())
        
        notes = self.notesPlayingWhile(pitchOrigin, True)
        allOctaveNotes = allOctaves(notes, 1, 6)
        
        ##Find where the current note pitch is in the chord progression list
        found = False
        iterations = 0
        if direction is 'descending':
            allOctaveNotes.reverse()
        
        for p in allOctaveNotes:
            if pitchOrigin.pitch < p:
                found = True
                iterations += 1
            if found and iterations >= stepSize:
                return p
            
        #If none found, return random note
        return random.choice(notes)

    def notesPlayingAtOffset(self, offset, sort = True):
    	"""
    	Gets notes playing from a given offset
    	"""
        n = note.Note()
        n.offset = offset
        return self.notesPlayingWhile(n, sort)

class RhythmManager(object):
	"""
	A rhythm manager is similar to a chord progression, but dealing with rhythms. Music21 streams are given as a 
	rhythm template. The manager class then provides a note factory wherein the generated notes conform to the rhythm
	of the stream with which the class is initialised. 

	As with the chord progression class, the rhythm manager internally loops the rhythm to any arbitrary length. 
	"""
    def __init__(self, initRhythm, constantSize = -1):        
    	"""
    	Args:
    		initRhythm (stream): stream to use as rhythmm, or None if generating constant rhythm
    		constantSize (float): optional -- if provided will create a constantly repeating rhythm with each note lasting constantSize quarter lengths.
    	"""
        self.rhythmStream = initRhythm
        self.currNote = 0
        self.baseOffset = 0
        
        if constantSize is not -1:
            ##Generate rhythm using constant length
            currOffset = 0.0
            p = stream.Stream()
            for i in range(0, 32):
                newNote = note.Note()
                newNote.duration.quarterLength = constantSize
                newNote.midi = 60
                p.insert(currOffset, newNote)
                currOffset += constantSize
            self.rhythmStream = p
      
    def next(self, moveToNext=True):
    	"""
    	Returns the next note in the rhythm
    	"""
        i = 0
        ##So this is O(N) every time we want the new note. Refactor if speed becomes an issue
        for n in self.rhythmStream:
            if type(n) is note.Note:
                if i is self.currNote:
                    if moveToNext:
                        self.currNote += 1
                    nn = note.Note()
                    nn.midi = 60
                    nn.duration.quarterLength = n.duration.quarterLength
                    nn.offset = n.offset + self.baseOffset
                    if n.volume.velocity is not None:
                        nn.volume.velocity = n.volume.velocity
                    return nn
                i += 1
                
        ##If we get to here, we've gone through the whole stream from the current index and found
        #only rests, so start again from the start
        
        ##Update internal counters
        self.rhythmStream.flat
        self.currNote = 0
        self.baseOffset = self.baseOffset + self.rhythmStream.duration.quarterLength
        return self.next()
      
    def reset(self, newBaseLength = 0):
    	"""
    	Resets the rhythm manager
    	"""
        self.baseOffset = newBaseLength
        self.currNote = 0
    


def allOctaves(noteList, lowOctave, highOctave):
	"""
	Takes a list of notes/scale and returns a list with those notes in every octave
	between lowOctave and highOctave
	"""
    if type(noteList) is scale.Scale:
        return noteList.getPitches("A"+str(lowOctave), "G#"+str(highOctave))
    
    toReturn = []
    for i in range(lowOctave, highOctave):
        for n in noteList:
            newNote = copy.deepcopy(n)
            newNote.octave = i
            toReturn.append(newNote)
    return toReturn

def convertChordsToNotes(s):
	"""
	Takes a stream and converts all the chords into sets of notes with identical offset
	"""
    for c in s.getElementsByClass(chord.Chord):
        for p in c.pitches:
            n = note.Note()
            n.duration = c.duration
            n.pitch = p
            s.insert(c.offset, n)
        s.remove(c)
    return s

def midiSort(obj):
	"""
	Used for sorting of note lists by pitch
	"""
    return obj.midi

def generateVoicingCandidates(ch, minOctave, maxOctave, maxNumber):
	"""
	Generates a set of possible candidate chord voicings in the range of given octaves
	around the given chord. Huge comp. complexity but ok since we don't
	use many notes in chords. If we did, this would not work very well...

	Todo:
		* Refactor to a less insane algorithm
	"""
    candidates = []
    pitchList = ch.pitches
    notesPerChord = len(pitchList)
    allPossiblePitches = allOctaves(pitchList, minOctave, maxOctave)
    for i in range(0,maxNumber):
        newChordList = []
        for b in range(0, notesPerChord):
            valid = False
            candidateNote = None
            loopProtection = 0
            while valid is False and loopProtection < 50:
                candidateNote = random.choice(allPossiblePitches)
                loopProtection += 1
                ##
                # Make sure its of a different pitch class to anything in the chord already
                valid = True
                for n in newChordList:
                    if n.midi%12 == candidateNote.midi % 12:
                        valid = False
            
            newChordList.append(candidateNote)
            
        candidates.append(chord.Chord(newChordList))
        
    return candidates

def psychoAcousticDistance(chord1, chord2):
	"""
	Calculates the psychoacoustic "distance" between two chords -- that is, how distant we perceive two
	chords to be. Minimising psychoacoustic distance often results in good voice-leading. 
	"""
    return euclidean(distRep(chord1, 0, 88), distRep(chord2, 0, 88))

def rho(h):
    return math.pow(0.7, h)

def M(f):
    return 12*(math.log(f,2) - math.log(8.1758,2))
 
def l(n, baseNote):
	"""
	Given a note, returns a perceived loudness of that note with respect to a baseNote -- i.e. the amplitude of the harmonic of baseNote
	that has the same frequency as the given note, if any.
	"""
    currMax = 0
    for h in range(1,50):
        freq = harmonicFrequency(baseNote, h)
        mf = M(freq)
        roundedMf = round(mf)
        if int(roundedMf) is n.midi:
            currMax = max([rho(h), currMax])
    return currMax
    
def frequency(m21note):
	"""
	Returns the frequency of the given note
	"""
    return m21note.pitch.frequency

def harmonicFrequency(m21note, h):
	"""
	Returns the frequency of the given h'th harmonic of a note. 
	"""
    return h * frequency(m21note)

def distRep(chor, x, y):
	"""
	Builds a distributed representation of the chord across the range
	given by x and y such that distRep[c,x,y] = { l(i, c_d) for all x <= i <= y and for all notes c_d in c }
	x and y are midi note numbers
	"""
    rep = dict()
    for frequencyNote in range(x,y):
        currentHarmonicNote = note.Note()
        currentHarmonicNote.midi = frequencyNote
        totalAmplitude = 0.0
        for noteP in chor.pitches:
            n = note.Note(noteP)
            totalAmplitude += l(currentHarmonicNote, n)
        rep[frequencyNote] = totalAmplitude
    return rep

def euclidean(x,y):
	"""
	Returns euclidean distance between two lists
	"""
    sumSq=0.0
    #add up the squared differences
    for i in range(len(x)):
            sumSq+=(x[i]-y[i])**2
    #take the square root of the result
    return (sumSq**0.5)

