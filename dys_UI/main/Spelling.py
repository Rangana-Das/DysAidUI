#Spello and FastPunct

from spello.model import SpellCorrectionModel
from fastpunct import FastPunct

class spelling:
	def __init__(self):
		self.text=None

	def correct(self):
		sp = SpellCorrectionModel(language='en')
		sp.load('en_large.pkl')
		f=open("../data/recog.txt",'r')
		self.text=(sp.spell_correct(f.readline())['spell_corrected_text'])
		print(self.text)

	def punct(self):
		fastpunct = FastPunct('en')
		x=(fastpunct.punct([self.text], batch_size=32))
		print(x)
		return x