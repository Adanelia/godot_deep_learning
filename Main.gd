extends Node


const dataset_path = "user://mnist_784.arff"
const dataset_url = "https://api.openml.org/data/download/52667/mnist_784.arff"
const save_network_path = "user://network.json"

var dataset = {}
var network# = {}
var train_index = 0
var test_index = 0

#var iters_num = 10000  # 适当设定循环的次数
var train_size = 0
#var batch_size = 100
#var learning_rate = 0.1

onready var http_request = $HTTPRequest
onready var texture_rect = $GUI/HSplitContainer/CenterContainer/ViewportContainer/Viewport/TextureRect
onready var dataset_label = $GUI/HSplitContainer/VBoxContainer/DatasetLabel
onready var download_button = $GUI/HSplitContainer/VBoxContainer/DownloadButton
onready var model_label = $GUI/HSplitContainer/VBoxContainer/ModelLabel
onready var load_model_button = $GUI/HSplitContainer/VBoxContainer/LoadModelButton
onready var show_train_button = $GUI/HSplitContainer/VBoxContainer/ShowTrainButton
onready var show_test_button = $GUI/HSplitContainer/VBoxContainer/ShowTestButton
onready var spinbox_from = $GUI/HSplitContainer/VBoxContainer/HBoxContainer/SpinBoxFrom
onready var spinbox_to = $GUI/HSplitContainer/VBoxContainer/HBoxContainer/SpinBoxTo
onready var spinbox_batch_size = $GUI/HSplitContainer/VBoxContainer/HBoxContainer2/SpinBoxBatchSize
onready var spinbox_learning_rate = $GUI/HSplitContainer/VBoxContainer/HBoxContainer2/SpinBoxLearningRate
onready var train_button = $GUI/HSplitContainer/VBoxContainer/TrainButton
onready var draw_button = $GUI/HSplitContainer/VBoxContainer/DrawButton
onready var predict_button = $GUI/HSplitContainer/VBoxContainer/PredictButton
onready var data_label = $GUI/HSplitContainer/VBoxContainer/GridContainer/DataLabel
onready var prediction_label = $GUI/HSplitContainer/VBoxContainer/GridContainer/PredictionLabel
onready var draw_canvas = $GUI/HSplitContainer/CenterContainer/ViewportContainer/Viewport/Draw
onready var draw_viewport = $GUI/HSplitContainer/CenterContainer/ViewportContainer/Viewport


class TwoLayerNet:
	
	var params = {}
	
	
	func _init(input_size : int, hidden_size : int, output_size : int, weight_init_std : float = 0.01):
		var rng = RandomNumberGenerator.new()
		rng.randomize()
		
		var w1 = []
		for i in input_size:
			w1.append([])
			for j in hidden_size:
				w1[i].append(rng.randfn() * weight_init_std)
		params["W1"] = Math.Matrix.new(w1)
		
		var b1 = []
		for i in hidden_size:
			b1.append(0.0)
		params["b1"] = Math.Matrix.new([b1])
		
		var w2 = []
		for i in hidden_size:
			w2.append([])
			for j in output_size:
				w2[i].append(rng.randfn() * weight_init_std)
		params["W2"] = Math.Matrix.new(w2)
		
		var b2 = []
		for i in output_size:
			b2.append(0.0)
		params["b2"] = Math.Matrix.new([b2])
	
	
	func predict(x : Math.Matrix) -> Math.Matrix:
		var a1 = x.dot(params["W1"]).add(params["b1"])
		var z1 = Math.sigmoid(a1)
		var a2 = z1.dot(params["W2"]).add(params["b2"])
		var y = Math.softmax(a2)
		
		return y
	
	
	func loss(x : Math.Matrix, t : Math.Matrix) -> float:
		var y = predict(x)
		
		return Math.cross_entropy_error(y, t)
	
	
	func accuracy_one_hot(x : Math.Matrix, t : Math.Matrix) -> float:
		var shape = x.shape()
		var y = predict(x).argmax(1)
		var tmp = t.argmax(1)
		
		var acc = 0
		for i in shape[1]:
			for j in shape[0]:
				if y.data[i][j] == tmp.data[i][j]:
					acc += 1
		acc /= shape[1]
		
		return acc
	
	
	func accuracy(x : Math.Matrix, t : Math.Matrix) -> float:
		var shape = x.shape()
		var y = predict(x).argmax(1)
		
		var acc = 0.0
		for i in shape[1]:
			if y.data[i][t.data[i][0]] == 1:
				acc += 1
		acc /= shape[1]
		
		return acc
	
	
	func gradient(x : Math.Matrix, t : Math.Matrix) -> Dictionary:
		var grads = {}
		var batch_num = x.shape()[1]
		
		# forward
		var a1 = x.dot(params["W1"]).add(params["b1"])
		var z1 = Math.sigmoid(a1)
		var a2 = z1.dot(params["W2"]).add(params["b2"])
		var y = Math.softmax(a2)
		
		# backward
		var shape_y = y.shape()
		var batch_num_array = []
		var batch_num_array_row = []
		batch_num_array_row.resize(shape_y[0])
		for i in shape_y[0]:
			batch_num_array_row[i] = batch_num
		for i in shape_y[1]:
			batch_num_array.append(batch_num_array_row)
		var batch_num_matrix = Math.Matrix.new(batch_num_array)
		var dy = y.sub(t).div(batch_num_matrix)
		grads["W2"] = z1.T().dot(dy)
		grads["b2"] = Math.Matrix.new([dy.sum(0).data[0]])
		
		var da1 = dy.dot(params["W2"].T())
		var dz1 = Math.sigmoid_grad(a1).mul(da1)
		grads["W1"] = x.T().dot(dz1)
		grads["b1"] = Math.Matrix.new([dz1.sum(0).data[0]])
		
		return grads
	
	
	func tojson() -> String:
		var json_dic = {}
		
		json_dic["W1"] = params["W1"].data
		json_dic["b1"] = params["b1"].data
		json_dic["W2"] = params["W2"].data
		json_dic["b2"] = params["b2"].data
		
		return to_json(json_dic)
	
	
	func fromjson(json_str : String):
		var json_dic = parse_json(json_str)
		
		params["W1"] = Math.Matrix.new(json_dic["W1"])
		params["b1"] = Math.Matrix.new(json_dic["b1"])
		params["W2"] = Math.Matrix.new(json_dic["W2"])
		params["b2"] = Math.Matrix.new(json_dic["b2"])


# Called when the node enters the scene tree for the first time.
func _ready():
	var file = File.new()
	if file.file_exists(dataset_path):
		download_button.disabled = true
		dataset_label.text = "Found dataset: " + dataset_path
		dataset = load_mnist()
	if file.file_exists(save_network_path):
		model_label.text = "Found model: " + save_network_path
		load_model_button.disabled = false


func load_mnist():
	print("Loading mnist...")
	var data = {"train": [], "test": []}
	
	var file = File.new()
	file.open(dataset_path, File.READ)
#	var content = file.get_as_text()
	var line = file.get_line()
	while line != "@DATA":
		line = file.get_line()
		if file.eof_reached():
			printerr("No data was found, is the dataset file correct?")
			file.close()
			return null
	
	# Load
	var count = 0
	line = file.get_csv_line()
	while not file.eof_reached() and count < 60000:# ERROR: All memory pool allocations are in use.
		if line.empty():
			continue
		
		printraw("\r", count)
#		print(line)
#		var line_int = PoolByteArray()
#		for i in line:
#			line_int.append(int(i))# Too slow!
#		var label = line_int[line_int.size() - 1]
#		line_int.remove(line_int.size() - 1)
#		var image_data = line_int
		
		var label = line[line.size() - 1]
		line.remove(line.size() - 1)
		
		if count < 10000:
#			data["test"].append({"label": label, "image": image_data})
			data["test"].append({"label": label, "image": line})
		else:
#			data["train"].append({"label": label, "image": image_data})
			data["train"].append({"label": label, "image": line})
		
		line = file.get_csv_line()
		count += 1
	
	file.close()
	train_size = data["train"].size()
	print()
	print("Finish loading.")
	
	return data


func _on_DownloadButton_pressed():
	download_button.text = "Downloading..."
	download_button.disabled = true
#	http_request.request("http://api.qingyunke.com/api.php?key=free&appid=0&msg=你好")
#	http_request.request("http://api.qingyunke.com/api.php?key=free&appid=0&msg=早")
	var error = http_request.request(dataset_url)
	if error != OK:
		push_error("An error occurred in the HTTP request.")
		download_button.text = "Download Dataset"
		download_button.disabled = false


func _on_HTTPRequest_request_completed(result, response_code, headers, body):
#	print(body)
#	print(body.get_string_from_utf8())
	var file = File.new()
	file.open(dataset_path, File.WRITE)
	file.store_buffer(body)
	file.close()
	download_button.text = "Completed"
	dataset = load_mnist()


func _on_OpenDataButton_pressed():
	OS.shell_open(ProjectSettings.globalize_path(dataset_path).get_base_dir())


func _on_LoadModelButton_pressed():
	network = TwoLayerNet.new(784, 50, 10)
	
	var file = File.new()
	file.open(save_network_path, File.READ)
	var json_str = file.get_as_text()
	file.close()
	
	network.fromjson(json_str)


func _on_ShowTrainButton_pressed():
	var texture = ImageTexture.new()
	var img = Image.new()
	var img_data = PoolByteArray()
	for i in dataset["train"][train_index]["image"]:
		img_data.append(i)
	img.create_from_data(28, 28, false, Image.FORMAT_L8, img_data)
	texture.create_from_image(img)
	texture_rect.texture = texture
	data_label.text = dataset["train"][train_index]["label"]
	prediction_label.text = ""
	train_index += 1
	show_train_button.text = "Next Training Image"


func _on_ShowTestButton_pressed():
	var texture = ImageTexture.new()
	var img = Image.new()
	var img_data = PoolByteArray()
	for i in dataset["test"][test_index]["image"]:
		img_data.append(i)
	img.create_from_data(28, 28, false, Image.FORMAT_L8, img_data)
	texture.create_from_image(img)
	texture_rect.texture = texture
	data_label.text = dataset["test"][test_index]["label"]
	prediction_label.text = ""
	test_index += 1
	show_test_button.text = "Next Testing Image"


func _on_ClearButton_pressed():
	texture_rect.texture = null


func _on_TrainButton_pressed():
	if network == null:
		network = TwoLayerNet.new(784, 50, 10)
	var batch_size = spinbox_batch_size.value
	var iters_from = spinbox_from.value
	var iters_to = spinbox_to.value
	var learning_rate = spinbox_learning_rate.value
#	var iter_per_epoch = max(train_size / batch_size, 1)
	
	for i in range(iters_from, iters_to):
		var idx = i % dataset["train"].size()
		var x_batch_data = dataset["train"].slice(batch_size * idx, batch_size * (idx + 1))
		var x_batch_array = []
		var t_batch_array = []
		for j in x_batch_data.size():
			var x_batch_array_row = []
			x_batch_array_row.resize(x_batch_data[j]["image"].size())
			for k in x_batch_data[j]["image"].size():
				x_batch_array_row[k] = int(x_batch_data[j]["image"][k])
			x_batch_array.append(x_batch_array_row)
			t_batch_array.append(int(x_batch_data[j]["label"]))
		var x_batch = Math.Matrix.new(x_batch_array)
		var t_batch = Math.Matrix.new([t_batch_array])
		
		var grad = network.gradient(x_batch, t_batch)
		
		var learning_rate_matrix = Math.Matrix.new([[learning_rate]])
		network.params["W1"] = network.params["W1"].sub(grad["W1"].mul(learning_rate_matrix))
		network.params["b1"] = network.params["b1"].sub(grad["b1"].mul(learning_rate_matrix))
		network.params["W2"] = network.params["W2"].sub(grad["W2"].mul(learning_rate_matrix))
		network.params["b2"] = network.params["b2"].sub(grad["b2"].mul(learning_rate_matrix))


func _on_AccuracyButton_pressed():
	if network == null:
		printerr("No data!")
		return
	
	dataset["test"].shuffle()
	var x_test_array = []
	var t_test_array = []
	
	for i in 100:#dataset["test"].size():
		var test_data = dataset["test"][i]
		var x_test_array_row = []
		for j in test_data["image"]:
			x_test_array_row.append(int(j))
		x_test_array.append(x_test_array_row)
		t_test_array.append([int(test_data["label"])])
	
	var x_test = Math.Matrix.new(x_test_array)
	var t_test = Math.Matrix.new(t_test_array)
#	var train_acc = network.accuracy(x_train, t_train)
	var test_acc = network.accuracy(x_test, t_test)
#	print("train accuracy: ", train_acc)
	print("test accuracy: ", test_acc)
	prediction_label.text = str(test_acc * 100) + "% test accuracy"


func _on_SaveButton_pressed():
	if network == null:
		printerr("Null")
		return
	var file = File.new()
	file.open(save_network_path, File.WRITE)
	file.store_string(network.tojson())
	file.close()


func _on_DrawButton_pressed():
	if draw_canvas.drawing:
		draw_button.text = "Draw"
		draw_canvas.stop_draw()
	else:
		draw_button.text = "Stop"
		draw_canvas.start_draw()


func _on_PredictButton_pressed():
	if network == null:
		printerr("No data!")
		return
	
	var img = draw_viewport.get_texture().get_data()
	img.flip_y()
#	print(img_data.data)
	img.convert(Image.FORMAT_L8)
	
	var result = network.predict(Math.Matrix.new([img.get_data()]))
	print(result.data)
	for each in result.data:
		var max_index = -1
		var max_num = 0
		for i in each.size():
			if max_num < each[i]:
				max_index = i
				max_num = each[i]
		var prediction_text = str(max_index) + " (" + str(stepify(max_num * 100, 0.01)) + "% confidence)"
		print(prediction_text)
		data_label.text = ""
		prediction_label.text = prediction_text
