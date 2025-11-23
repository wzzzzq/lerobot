#!/usr/bin/env python3
"""比较teacher和student的config"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

teacher_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin/checkpoints/last/pretrained_model"
student_path = "/pfs/pfs-ilWc5D/ziqianwang/new_pretrain/put_bottles_dustbin_reflow_new/checkpoints/last/pretrained_model"

print("Loading models...")
teacher = SmolVLAPolicy.from_pretrained(teacher_path)
student = SmolVLAPolicy.from_pretrained(student_path)

print('\n' + '=' * 60)
print('TEACHER CONFIG:')
print('=' * 60)
print(f'num_steps: {teacher.config.num_steps}')
print(f'n_obs_steps: {teacher.config.n_obs_steps}')
print(f'n_action_steps: {teacher.config.n_action_steps}')
print(f'action_chunk_size: {teacher.config.action_chunk_size}')
print(f'max_action_dim: {teacher.config.max_action_dim}')
print(f'action_feature: {teacher.config.action_feature}')

print()
print('=' * 60)
print('STUDENT CONFIG:')
print('=' * 60)
print(f'num_steps: {student.config.num_steps}')
print(f'n_obs_steps: {student.config.n_obs_steps}')
print(f'n_action_steps: {student.config.n_action_steps}')
print(f'action_chunk_size: {student.config.action_chunk_size}')
print(f'max_action_dim: {student.config.max_action_dim}')
print(f'action_feature: {student.config.action_feature}')

print()
print('=' * 60)
print('DIFFERENCES:')
print('=' * 60)
diff_found = False
if teacher.config.num_steps != student.config.num_steps:
    print(f'⚠️  num_steps: teacher={teacher.config.num_steps}, student={student.config.num_steps}')
    diff_found = True
if teacher.config.n_obs_steps != student.config.n_obs_steps:
    print(f'⚠️  n_obs_steps: teacher={teacher.config.n_obs_steps}, student={student.config.n_obs_steps}')
    diff_found = True
if teacher.config.n_action_steps != student.config.n_action_steps:
    print(f'⚠️  n_action_steps: teacher={teacher.config.n_action_steps}, student={student.config.n_action_steps}')
    diff_found = True
if teacher.config.action_chunk_size != student.config.action_chunk_size:
    print(f'⚠️  action_chunk_size: teacher={teacher.config.action_chunk_size}, student={student.config.action_chunk_size}')
    diff_found = True

if not diff_found:
    print("✓ No differences found in key configs")
