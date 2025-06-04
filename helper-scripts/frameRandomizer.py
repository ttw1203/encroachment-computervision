import os
import random
import shutil
from pathlib import Path
from collections import defaultdict
import math  # Not strictly necessary for round(), but good practice if other math functions were used

# --- Configuration ---
# Input image folder (as per your context)
image_dir = Path("D:\\thesisVideo\\croppedROI")
# Output base directory (as per your context)
base_out_dir = Path("D:\\thesisVideo\\croppedROI\\randomized")
splits = ['train', 'val', 'test']
# Desired ratios for train, val, test
ratios = [0.7, 0.15, 0.15]
# Random seed for reproducibility
random.seed(42)


# --- Main Script ---

def create_splits():
    # Create output folders for train, val, test
    for split_name in splits:
        (base_out_dir / split_name).mkdir(parents=True, exist_ok=True)

    # 1. Group images by prefix
    groups = defaultdict(list)
    print(f"Scanning images in: {image_dir}")
    image_files = list(image_dir.glob("*.jpg"))
    if not image_files:
        print(f"❌ No JPG images found in {image_dir}. Please check the path.")
        return

    for img_path in image_files:
        fname = img_path.name
        prefix = "unknown"  # Default prefix
        if "banani" in fname or "07." in fname:
            prefix = "banani"
        elif "bns" in fname:
            prefix = "bns"
        elif "boardbazar" in fname:
            prefix = "boardbazar"
        elif "mirpur1" in fname or "mirpur2" in fname:  # Handles both mirpur1 and mirpur2
            prefix = "mirpur"
        # 'unknown' remains if none of the above conditions are met
        groups[prefix].append(img_path)

    if not groups:
        print("❌ No image groups were formed. Ensure filenames match expected patterns.")
        return

    print(f"Found {len(groups)} unique groups: {list(groups.keys())}")
    for prefix, file_list in groups.items():
        print(f"  Group '{prefix}': {len(file_list)} images")
    print("-" * 30)

    # 2. Randomly assign whole groups to splits
    group_keys = list(groups.keys())
    random.shuffle(group_keys)
    total_num_groups = len(group_keys)

    assigned_group_splits = {}  # To store which split each group_key belongs to
    group_counts_per_split = {'train': 0, 'val': 0, 'test': 0}

    # 3. Guarantee non-empty val and test (adjust counts gracefully)
    if total_num_groups == 0:
        print("❌ No groups to split.")
        return
    elif total_num_groups == 1:
        # Assign the single group to train
        assigned_group_splits[group_keys[0]] = 'train'
        group_counts_per_split['train'] = 1
    elif total_num_groups == 2:
        # Assign one to train, one to val
        assigned_group_splits[group_keys[0]] = 'train'
        assigned_group_splits[group_keys[1]] = 'val'
        group_counts_per_split['train'] = 1
        group_counts_per_split['val'] = 1
    else:  # total_num_groups >= 3
        # Calculate target counts, ensuring val and test get at least 1
        val_count = max(1, round(ratios[1] * total_num_groups))
        test_count = max(1, round(ratios[2] * total_num_groups))

        # Remaining groups go to train. Ensure train doesn't become negative if val/test rounded up significantly.
        train_count = total_num_groups - val_count - test_count

        # If train_count becomes less than 1 due to val_count + test_count >= total_num_groups
        # (e.g. if N=3, val_count=1, test_count=1, then train_count=1, which is fine)
        # (e.g. if N=3, but ratios forced val_count=2, test_count=2, then train_count=-1 - this shouldn't happen with sum(ratios)=1 and round())
        # However, to be absolutely safe, if train_count is pushed low:
        if train_count < 1:  # This implies val_count + test_count >= total_num_groups
            # This scenario is rare if N>=3 and max(1,round()) is used.
            # Most likely val_count + test_count == total_num_groups, so train_count = 0.
            # Force train to have at least 1 if it became 0, by taking from the largest of val/test if they are > 1
            if train_count < 1 and total_num_groups >= 3:  # Ensure train gets at least 1 for N>=3
                train_count = 1
                # Reduce val or test if sum now exceeds total_num_groups
                if (val_count + test_count + train_count > total_num_groups):
                    if val_count > test_count and val_count > 1:
                        val_count -= 1
                    elif test_count > 1:
                        test_count -= 1
                    # If both were 1, and train is 1, then sum must be N (e.g. 1,1,1 for N=3)

        # Distribute any discrepancy due to rounding (if sum is not total_num_groups)
        # This is typically handled by assigning train_count last as the remainder.
        # Re-calculate train_count as the definite remainder to ensure sum is correct.
        current_sum = val_count + test_count
        if current_sum > total_num_groups:  # If val and test alone are too many
            # This implies N=2 case was almost hit by N=3 with aggressive rounding.
            # e.g. N=3, ratio_val=0.4, ratio_test=0.4 -> val=round(1.2)=1, test=round(1.2)=1. Total 2.
            # Fallback to N=3 -> 1,1,1
            if total_num_groups == 3:
                train_count, val_count, test_count = 1, 1, 1
            else:  # Should not happen with proper ratio sum and N>=3
                # Prioritize train, then val, then test for allocation
                train_count = round(ratios[0] * total_num_groups)
                val_count = round(ratios[1] * total_num_groups)
                test_count = total_num_groups - train_count - val_count
                # And re-ensure minimums if N>=3
                if val_count == 0: val_count = 1; train_count = max(0, train_count - 1)
                if test_count == 0: test_count = 1; train_count = max(0, train_count - 1)
                if train_count == 0 and total_num_groups - (
                        val_count + test_count) > 0:  # if train got emptied ensure it gets rest
                    train_count = total_num_groups - (val_count + test_count)

        # Final assignment of counts after adjustments
        # The most robust for N>=3:
        group_counts_per_split['val'] = max(1, round(ratios[1] * total_num_groups))
        group_counts_per_split['test'] = max(1, round(ratios[2] * total_num_groups))
        group_counts_per_split['train'] = total_num_groups - group_counts_per_split['val'] - group_counts_per_split[
            'test']

        # If train count became < 1 (e.g. 0 or negative) after val and test took their `max(1, round())` share
        if group_counts_per_split['train'] < 1:
            # This means val and test (both >=1) sum up to N or N-1.
            # e.g. N=3, val=1, test=1 -> train=1.
            # e.g. N=2 -> handled.
            # If N=3, val_ratio=0.5, test_ratio=0.5 -> val=max(1,round(1.5))=2, test=max(1,round(1.5))=2. Train = 3-2-2 = -1. This is a bad ratio set.
            # Assuming ratios sum to 1:
            # N=3, r=[0.1,0.45,0.45]. val=max(1,round(1.35))=1. test=max(1,round(1.35))=1. train=3-1-1=1. Correct.
            # The logic `train = N - val - test` ensures sum is N.
            # And `val >= 1`, `test >= 1`. So `train <= N-2`.
            # If N=3, train <= 1. If train is 1, it's good. If train is <1 (i.e. 0 or negative),
            # it means N - val - test < 1.
            # For N=3, val=1, test=1, then train=1. No issue.
            # This readjustment ensures train gets at least 1 if N >= val_min + test_min + train_min (i.e. 1+1+1=3)
            if total_num_groups - (
                    group_counts_per_split['val'] + group_counts_per_split['test']) < 1 and total_num_groups >= 1:
                # This indicates that val and test took all or nearly all. We need to rebalance to give train its share.
                # This situation arises if sum of ratios for val and test is high.
                # Safest for N>=3 is to ensure all are at least 1.
                # Recalculate train, then re-distribute remainder if sum of calculated counts != total_num_groups
                temp_train = round(ratios[0] * total_num_groups)
                temp_val = round(ratios[1] * total_num_groups)
                temp_test = total_num_groups - temp_train - temp_val  # Remainder for test

                # Check if any became negative after this (e.g. if train+val > total)
                if temp_test < 0:  # if test is negative, val was too big
                    temp_test = 0
                    temp_val = total_num_groups - temp_train
                if temp_val < 0:  # if val is negative, train was too big
                    temp_val = 0
                    temp_train = total_num_groups - temp_test

                group_counts_per_split['train'] = temp_train
                group_counts_per_split['val'] = temp_val
                group_counts_per_split['test'] = temp_test

                # Enforce minimums of 1 for val and test if total_num_groups >= 3
                # And ensure train gets at least 1 if possible
                if group_counts_per_split['val'] == 0 and total_num_groups >= 2:  # val needs at least 1 if 2+ groups
                    group_counts_per_split['val'] = 1
                    if group_counts_per_split['train'] > 0:
                        group_counts_per_split['train'] -= 1
                    elif group_counts_per_split['test'] > 0:
                        group_counts_per_split['test'] -= 1  # Should not happen with train priority

                if group_counts_per_split['test'] == 0 and total_num_groups >= 3:  # test needs at least 1 if 3+ groups
                    group_counts_per_split['test'] = 1
                    if group_counts_per_split['train'] > 0:
                        group_counts_per_split['train'] -= 1
                    elif group_counts_per_split['val'] > 1:
                        group_counts_per_split['val'] -= 1  # Steal from val if val has more than 1

                # Correct any sum mismatch by adjusting train
                current_sum_final = group_counts_per_split['train'] + group_counts_per_split['val'] + \
                                    group_counts_per_split['test']
                diff = total_num_groups - current_sum_final
                group_counts_per_split['train'] += diff

        # Assign groups to splits based on calculated counts
        current_idx = 0
        for i in range(group_counts_per_split['train']):
            if current_idx < total_num_groups:
                assigned_group_splits[group_keys[current_idx]] = 'train'
                current_idx += 1
        for i in range(group_counts_per_split['val']):
            if current_idx < total_num_groups:
                assigned_group_splits[group_keys[current_idx]] = 'val'
                current_idx += 1
        for i in range(group_counts_per_split['test']):
            if current_idx < total_num_groups:
                assigned_group_splits[group_keys[current_idx]] = 'test'
                current_idx += 1

        # Ensure all groups are assigned if counts were off due to complex adjustments
        if current_idx < total_num_groups:  # Some groups left unassigned
            print(
                f"Warning: {total_num_groups - current_idx} groups unassigned due to count calculation issues. Assigning remaining to 'train'.")
            while current_idx < total_num_groups:
                assigned_group_splits[group_keys[current_idx]] = 'train'
                group_counts_per_split['train'] += 1  # adjust count
                current_idx += 1

    # 4. Copy images into split folders
    image_counts_per_split = defaultdict(int)
    for group_prefix, image_path_list in groups.items():
        if group_prefix not in assigned_group_splits:
            print(f"Warning: Group '{group_prefix}' was not assigned a split. Skipping {len(image_path_list)} images.")
            continue

        split_assignment = assigned_group_splits[group_prefix]
        output_folder_for_group = base_out_dir / split_assignment

        for img_path in image_path_list:
            try:
                shutil.copy(img_path, output_folder_for_group / img_path.name)
                image_counts_per_split[split_assignment] += 1
            except Exception as e:
                print(f"Error copying {img_path} to {output_folder_for_group}: {e}")

    # 5. Print summary
    print("-" * 30)
    print("✅ Image grouping and split completed.")
    print("Summary:")
    print(
        f"  Groups → train: {group_counts_per_split['train']}, val: {group_counts_per_split['val']}, test: {group_counts_per_split['test']}")
    print(f"  Total groups: {total_num_groups}")
    print("  Images per split:")
    print(f"    train: {image_counts_per_split['train']} images")
    print(f"    val:   {image_counts_per_split['val']} images")
    print(f"    test:  {image_counts_per_split['test']} images")
    print(f"  Total images processed: {sum(image_counts_per_split.values())}")
    print("-" * 30)


if __name__ == '__main__':
    create_splits()