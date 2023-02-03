extends Control


const SCALE = 128.0 / 28.0

var drawing := false
var points = []
var mouse_down := false

var brush_color = Color(1.0, 1.0, 1.0)
var background_color = Color(0.0, 0.0, 0.0)


# Called when the node enters the scene tree for the first time.
func _ready():
	set_process_input(false)


func _input(event):
	if event is InputEventMouseButton:
		if event.is_pressed():
			mouse_down = true
			var vector2array = [get_local_mouse_position() / SCALE]#PoolVector2Array([get_local_mouse_position()])
			points.append(vector2array)
		else:
			mouse_down = false
#			print(points)
			update()
	if event is InputEventMouseMotion:
		if mouse_down:
			points[-1].append(get_local_mouse_position() / SCALE)


func _draw():
	if drawing:
		draw_background()
		draw_lines()


func draw_background():
	draw_rect(get_rect(), background_color)


func draw_lines():
	for vector2array in points:
		for i in vector2array.size() - 1:
			draw_line(vector2array[i], vector2array[i + 1], brush_color, 3.0)


func start_draw():
	drawing = true
	points.clear()
	set_process_input(true)
	update()


func stop_draw():
	set_process_input(false)
	drawing = false
	points.clear()
	update()

