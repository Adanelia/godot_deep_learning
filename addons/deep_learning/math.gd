class_name Math
extends Node


# Declare member variables here. Examples:
# var a = 2
# var b = "text"


class Matrix:
	
	var data : Array
	
	
	func _init(array : Array = []):
#		# Check
#		for row in array:
#			# Each row should be array
#			if typeof(row) != TYPE_ARRAY:
#				printerr("Type error.")
#				return null
#			for each in row:
#				# Each variable should be numerical
#				if typeof(each) != TYPE_INT or typeof(each) != TYPE_REAL or typeof(each) != TYPE_BOOL:
#					printerr("Type error.")
#					return null
#		# Each row should have the same length
#		var row_length = array[0].size()
#		for row in array:
#			if row.size() != row_length:
#				printerr("Type error.")
#				return null
#			row_length = row.size()
		
		data = array
	
	
	func duplicate() -> Matrix:
		return Matrix.new(data.duplicate(true))
	
	
	func add(matrix : Matrix) -> Matrix:
		var fix_x = matrix.shape()[1] == 1
		var fix_y = matrix.shape()[0] == 1
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var i2 = i
				var j2 = j
				if fix_x:
					i2 = 0
				if fix_y:
					j2 = 0
				var r = data[i][j] + matrix.data[i2][j2]
				result.data[i].append(r)
		
		return result
	
	
	func sub(matrix : Matrix) -> Matrix:
		var fix_x = matrix.shape()[1] == 1
		var fix_y = matrix.shape()[0] == 1
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var i2 = i
				var j2 = j
				if fix_x:
					i2 = 0
				if fix_y:
					j2 = 0
				var r = data[i][j] - matrix.data[i2][j2]
				result.data[i].append(r)
		
		return result
	
	
	func mul(matrix : Matrix) -> Matrix:
		var fix_x = matrix.shape()[1] == 1
		var fix_y = matrix.shape()[0] == 1
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var i2 = i
				var j2 = j
				if fix_x:
					i2 = 0
				if fix_y:
					j2 = 0
				var r = data[i][j] * matrix.data[i2][j2]
				result.data[i].append(r)
		
		return result
	
	
	func div(matrix : Matrix) -> Matrix:
		var fix_x = matrix.shape()[1] == 1
		var fix_y = matrix.shape()[0] == 1
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var i2 = i
				var j2 = j
				if fix_x:
					i2 = 0
				if fix_y:
					j2 = 0
				var r = data[i][j] / matrix.data[i2][j2]
				result.data[i].append(r)
		
		return result
	
	
	func exp() -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var r = exp(data[i][j])
				result.data[i].append(r)
		
		return result
	
	
	func log() -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[1]:
			result.data.append([])
			for j in shape[0]:
				var r = log(data[i][j])
				result.data[i].append(r)
		
		return result
	
	
	func shape():
		return [data[0].size(), data.size()]
	
	
	func dot(matrix : Matrix) -> Matrix:
		# Check
		var shape_a = shape()
		var shape_b = matrix.shape()
		if shape_a[0] != shape_b[1]:
			printerr("Shapes not aligned.")
			return null
		
		var result = Matrix.new()
		
		for i in shape_a[1]:
			result.data.append([])
			for j in shape_b[0]:
				var r = 0
				for k in shape_b[1]:
					r += matrix.data[k][j] * data[i][k]
				result.data[i].append(r)
		
		return result
	
	
	func max(axis : int = 1) -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		if axis == 1:
			for i in shape[1]:
				result.data.append([])
				var r = data[i][0]
				for j in shape[0]:
					if r < data[i][j]:
						r = data[i][j]
				for j in shape[0]:
					result.data[i].append(r)
		elif axis == 0:
			var r = data[0].duplicate()
			for i in shape[1]:
				for j in shape[0]:
					if r[j] < data[i][j]:
						r[j] = data[i][j]
			for i in shape[1]:
				result.data.append(r)
		
		return result
	
	
	func min(axis : int = 1) -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		if axis == 1:
			for i in shape[1]:
				result.data.append([])
				var r = data[i][0]
				for j in shape[0]:
					if r > data[i][j]:
						r = data[i][j]
				for j in shape[0]:
					result.data[i].append(r)
		elif axis == 0:
			var r = data[0].duplicate()
			for i in shape[1]:
				for j in shape[0]:
					if r[j] > data[i][j]:
						r[j] = data[i][j]
			for i in shape[1]:
				result.data.append(r)
		
		return result
	
	
	func sum(axis : int = 1) -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		if axis == 1:
			for i in shape[1]:
				result.data.append([])
				var r = 0
				for j in shape[0]:
					r += data[i][j]
				for j in shape[0]:
					result.data[i].append(r)
		elif axis == 0:
			var r = data[0].duplicate()
			for i in shape[1]:
				for j in shape[0]:
					r[j] += data[i][j]
			for i in shape[1]:
				result.data.append(r)
		
		return result
	
	
	func argmax(axis : int = 1) -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		if axis == 1:
			for i in shape[1]:
				result.data.append([])
				var index = 0
				var r = data[i][index]
				for j in shape[0]:
					if r < data[i][j]:
						index = j
						r = data[i][j]
				for j in shape[0]:
					if j == index:
						result.data[i].append(1)
					else:
						result.data[i].append(0)
		elif axis == 0:
			for i in shape[1]:
				result.data.append([])
			for j in shape[0]:
				var index = 0
				var r = data[index][j]
				for i in shape[1]:
					if r < data[i][j]:
						index = i
						r = data[i][j]
				for i in shape[1]:
					if i == index:
						result.data[i].append(1)
					else:
						result.data[i].append(0)
		
		return result
	
	
	func flatten() -> Matrix:
		var result = Matrix.new()
		
		result.data.append([])
		for i in data:
			result.data.append_array(i)
		
		return result
	
	
	func T() -> Matrix:
		var shape = shape()
		var result = Matrix.new()
		
		for i in shape[0]:
			var row = []
			row.resize(shape[1])
			result.data.append(row)
			for j in shape[1]:
				result.data[i][j] = data[j][i]
		
		return result


static func sigmoid(x : Matrix) -> Matrix:
	var shape = x.shape()
	var result = Matrix.new()
	
	for i in shape[1]:
		result.data.append([])
		for j in shape[0]:
			var r = 1 / (1 + exp(-x.data[i][j]))
			result.data[i].append(r)
	
	return result


static func sigmoid_grad(x : Matrix) -> Matrix:
	var shape = x.shape()
	var result = Matrix.new()
	
	for i in shape[1]:
		result.data.append([])
		for j in shape[0]:
			result.data[i].append(1.0)
	
	var tmp = sigmoid(x)
	result = result.sub(tmp).mul(tmp)
	
	return result


static func relu(x : Matrix) -> Matrix:
	var shape = x.shape()
	var result = Matrix.new()
	
	for i in shape[1]:
		result.data.append([])
		for j in shape[0]:
			var r = max(0, x.data[i][j])
			result.data[i].append(r)
	
	return result


static func softmax(x : Matrix) -> Matrix:
	var tmp = x.sub(x.max(1))
	return tmp.exp().div(tmp.exp().sum(1))


static func mean_squared_error(y : Matrix, t : Matrix) -> float:
	var tmp = y.sub(t)
	return 0.5 * tmp.mul(tmp).sum().data[0][0]


static func cross_entropy_error_one_hot(y : Matrix, t : Matrix) -> float:
	var tmp = y.duplicate()
	var batch_size = y.shape[1]
	for i in tmp.data:
		for j in i:
			j += 0.0000001
	
	return -t.mul(tmp.log()).sum().data[0][0] / batch_size


static func cross_entropy_error(y : Matrix, t : Matrix) -> float:
	var batch_size = y.shape[1]
	var r = 0
	for i in batch_size:
		r += log(y.data[i][t.data[i][0]] + 0.0000001)
	
	return -r / batch_size



