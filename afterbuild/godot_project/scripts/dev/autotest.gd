extends Node2D
## End-to-end test: drives one dialogue per NPC through the real client stack.
## Run: godot --path afterbuild/godot_project res://scenes/dev/autotest.tscn
## Never set as the project's main scene.

const NPCS := ["Leonardo", "Einstein", "Shakespeare", "Socrates"]
const PER_NPC_TIMEOUT_S := 60.0
const CONNECT_TIMEOUT_S := 10.0

var bar: Node = null
var current_npc := ""
var npc_index := 0
var token_count := 0
var elapsed := 0.0
var waiting := false
var connect_elapsed := 0.0
var failed := false


func _ready() -> void:
	bar = load("res://scenes/cozy_bar.tscn").instantiate()
	add_child(bar)
	print("AUTOTEST: waiting for websocket...")


func _process(delta: float) -> void:
	if failed:
		return
	if not waiting:
		connect_elapsed += delta
		if bar.is_websocket_connected:
			_start_next()
		elif connect_elapsed > CONNECT_TIMEOUT_S:
			_fail("no websocket after %ss" % CONNECT_TIMEOUT_S)
		return
	elapsed += delta
	if elapsed > PER_NPC_TIMEOUT_S:
		_fail("timeout for %s" % current_npc)


func _start_next() -> void:
	if npc_index >= NPCS.size():
		print("AUTOTEST ALL PASS")
		get_tree().quit(0)
		return
	if not bar.response_completed.is_connected(_on_response_completed):
		bar.response_completed.connect(_on_response_completed)
		bar.token_received.connect(_on_token_received)
	current_npc = NPCS[npc_index]
	npc_index += 1
	token_count = 0
	elapsed = 0.0
	waiting = true
	print("AUTOTEST: sending to ", current_npc)
	bar.interact_with_npc(current_npc, "Hello from autotest!")


func _on_token_received(npc: String, _token: String) -> void:
	if npc == current_npc:
		token_count += 1


func _on_response_completed(npc: String, full: String) -> void:
	if npc != current_npc:
		return
	waiting = false
	print("AUTOTEST PASS npc=%s tokens=%d ms=%d text=%s" % [npc, token_count, int(elapsed * 1000.0), full.substr(0, 80)])
	if token_count < 1:
		_fail("%s completed with no tokens" % npc)
	# The next NPC is started from _process on the next frame: the dialogue
	# client clears is_dialogue_processing only AFTER emitting this signal,
	# so calling interact_with_npc() here would be dropped as "still processing".


func _fail(reason: String) -> void:
	failed = true
	print("AUTOTEST FAIL: ", reason)
	get_tree().quit(1)
