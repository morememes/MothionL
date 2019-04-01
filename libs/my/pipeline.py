

class Predictor:
	def __init__(self, detector, net):
		self.detector = detector
		self.net = net
		#self.dataloader = None

	def predict(self, dataloader):
		"""
			1. Получить батч
				1.1 Препроцессинг батч
			2. Задетектить боксы в батче
				2.1 Препроцессинг батч
			3. Взять маску
				3.1 Постпроцессинг батч
		"""


		preds = []
		ret = True

		while ret:
			ret, batch = dataloader.__get_batch__()
			boxes, labels, probs = self.detector.predict(batch, 10, 0.4)



