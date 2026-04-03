import scipy.io as sio

# --- UPDATE THIS PATH ---
EVENTS_FILE_PATH = r'/Users/aryan/Downloads/SA09_20251013/bpod_session_data.mat'

mat_data = sio.loadmat(EVENTS_FILE_PATH, simplify_cells=True)

print("\n✅ Loaded file. Top-level keys:")
print(mat_data.keys())

SessionData = mat_data['SessionData']

print("\n✅ Keys inside SessionData:")
print(SessionData.keys())

print("\n✅ RawEvents structure for Trial 0:")
print(SessionData['RawEvents']['Trial'][0].keys())

print("\n✅ Event keys inside Trial 0:")
event_keys = SessionData['RawEvents']['Trial'][0]['Events'].keys()
print(event_keys)

print("\n✅ Event keys across ALL trials:")
all_keys = set()
for trial in SessionData['RawEvents']['Trial']:
    all_keys.update(trial['Events'].keys())

print(all_keys)
