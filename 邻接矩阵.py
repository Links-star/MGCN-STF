import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Circle


# Geometric Calculation Utility Class
class GeoUtils:
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two points (meters)"""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = radians(lat1), radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)

        a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    @staticmethod
    def calculate_radius_from_area(area):
        """Calculate circle radius from area (meters)"""
        return sqrt(area / np.pi)

    @staticmethod
    def meters_to_degrees(meters, latitude):
        """Convert meters to latitude/longitude degrees"""
        lat_deg = meters / 111319.5  # 1 degree latitude ≈ 111319.5 meters
        lon_deg = meters / (111319.5 * cos(radians(latitude)))  # Longitude varies with latitude
        return lat_deg, lon_deg


# Load single day data (with column name check)
def load_single_day_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Data file not found! Path: {file_path}")

    if not file_path.endswith(('.xlsx', '.xls')):
        raise ValueError("Error: Data file must be Excel format (.xlsx or .xls)")

    try:
        # Read all column names for verification
        all_columns = pd.read_excel(file_path, sheet_name='Sheet1', nrows=0).columns.tolist()
        print(f"\nActual column names in {os.path.basename(file_path)}:")
        print(all_columns)

        # Target columns (adjust based on actual Excel column names)
        target_columns = ['ID', 'Longitude', 'Latitude', 'Food_Category']

        # Check for missing columns
        missing_cols = [col for col in target_columns if col not in all_columns]
        if missing_cols:
            raise ValueError(f"File missing required columns: {missing_cols}")

        # Read file and add date identifier
        df = pd.read_excel(file_path, sheet_name='Sheet1', usecols=target_columns)

        # Extract date from filename (assuming format: "202108-1_Food_Delivery_Orders.xlsx")
        base_name = os.path.basename(file_path)
        date_str = base_name.split('_')[0].split('-')[-1]
        df['Date'] = f"202108{date_str}"

        # Clean data
        df = df.dropna(subset=['Latitude', 'Longitude'])
        print(f"Data loaded: {file_path}, valid records: {len(df)}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {str(e)}")


# Load multiple days of data
def load_multiple_days_data(file_paths):
    """Load and merge order data from multiple dates"""
    all_dfs = []
    for path in file_paths:
        df = load_single_day_data(path)
        all_dfs.append(df)

    # Merge all data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nAll data merged, total records: {len(combined_df)}")
    print(f"Date range: {sorted(combined_df['Date'].unique())}")
    return combined_df


# Identify dense data regions
def identify_dense_regions(df, eps=0.005, min_samples=30):
    """Identify dense data regions using DBSCAN"""
    coords = df[['Latitude', 'Longitude']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['dense_cluster'] = dbscan.fit_predict(coords)

    # Extract dense region centers
    dense_centers = []
    unique_clusters = set(df['dense_cluster'])
    if -1 in unique_clusters:
        unique_clusters.remove(-1)

    for cluster in unique_clusters:
        cluster_points = df[df['dense_cluster'] == cluster]
        center_lat = cluster_points['Latitude'].mean()
        center_lon = cluster_points['Longitude'].mean()
        dense_centers.append((center_lat, center_lon))

    print(f"Identified {len(dense_centers)} dense data regions")
    return df, dense_centers


# Generate balanced equal-area covering circles
def create_balanced_equal_area_circles(df, n_circles=30, overlap_ratio=0.4, dense_weight=1.5):
    """Generate balanced equal-area covering circles"""
    df, dense_centers = identify_dense_regions(df)
    n_dense = len(dense_centers)

    # Calculate data boundary range (exclude outliers with 5% quantile)
    lat_q1, lat_q3 = df['Latitude'].quantile([0.05, 0.95])
    lon_q1, lon_q3 = df['Longitude'].quantile([0.05, 0.95])

    min_lat, max_lat = lat_q1, lat_q3
    min_lon, max_lon = lon_q1, lon_q3
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Calculate core area
    width = GeoUtils.haversine_distance(center_lat, min_lon, center_lat, max_lon)
    height = GeoUtils.haversine_distance(min_lat, center_lon, max_lat, center_lon)
    total_area = width * height
    circle_area = (total_area * (1 + overlap_ratio)) / n_circles
    circle_radius_m = GeoUtils.calculate_radius_from_area(circle_area)
    radius_lat, radius_lon = GeoUtils.meters_to_degrees(circle_radius_m, center_lat)

    print(f"Core study area: {total_area:.2f} square meters")
    print(f"Each circle area: {circle_area:.2f} square meters")
    print(f"Each circle radius: {circle_radius_m:.2f} meters")

    # Allocate circle counts between dense and sparse regions
    if n_dense > 0:
        dense_circle_count = min(int(n_circles * (dense_weight / (dense_weight + 1))), n_circles - 5)
        sparse_circle_count = n_circles - dense_circle_count
        print(f"Allocated {dense_circle_count} circles to dense regions, {sparse_circle_count} to sparse regions")
    else:
        dense_circle_count = 0
        sparse_circle_count = n_circles

    # 1. Generate circles in dense regions
    dense_centers_generated = []
    if dense_circle_count > 0 and n_dense > 0:
        circles_per_dense = max(1, dense_circle_count // n_dense)
        remaining = dense_circle_count % n_dense

        for i, (lat, lon) in enumerate(dense_centers):
            for di in range(circles_per_dense + (1 if i < remaining else 0)):
                for dj in range(circles_per_dense + (1 if i < remaining else 0)):
                    if len(dense_centers_generated) >= dense_circle_count:
                        break

                    step_lat = radius_lat * 2 * (1 - overlap_ratio) * 0.6
                    step_lon = radius_lon * 2 * (1 - overlap_ratio) * 0.6

                    new_lat = lat + (di - circles_per_dense // 2) * step_lat
                    new_lon = lon + (dj - circles_per_dense // 2) * step_lon

                    if min_lat <= new_lat <= max_lat and min_lon <= new_lon <= max_lon:
                        dense_centers_generated.append((new_lat, new_lon))
                if len(dense_centers_generated) >= dense_circle_count:
                    break

    # 2. Generate circles in sparse regions
    sparse_centers_generated = []
    grid_step_lat = radius_lat * 2 * (1 - overlap_ratio)
    grid_step_lon = radius_lon * 2 * (1 - overlap_ratio)

    n_rows = int(np.ceil((max_lat - min_lat) / grid_step_lat)) + 1
    n_cols = int(np.ceil((max_lon - min_lon) / grid_step_lon)) + 1

    for i in range(n_rows):
        for j in range(n_cols):
            if len(sparse_centers_generated) >= sparse_circle_count:
                break

            lat = min_lat + i * grid_step_lat
            lon = min_lon + j * grid_step_lon
            if i % 2 == 1:
                lon += grid_step_lon / 2  # Hexagonal offset

            if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                too_close = False
                for (d_lat, d_lon) in dense_centers_generated:
                    if GeoUtils.haversine_distance(lat, lon, d_lat, d_lon) < circle_radius_m * 0.5:
                        too_close = True
                        break
                if not too_close:
                    sparse_centers_generated.append((lat, lon))
        if len(sparse_centers_generated) >= sparse_circle_count:
            break

    # Merge and ensure total count is n_circles
    all_centers = dense_centers_generated + sparse_centers_generated
    if len(all_centers) > n_circles:
        all_centers = all_centers[:n_circles]
    elif len(all_centers) < n_circles:
        all_centers += [all_centers[i % len(all_centers)] for i in range(n_circles - len(all_centers))]

    # Save circle information
    circles = {
        'centers': np.array(all_centers),
        'radius_m': circle_radius_m,
        'radius_lat': radius_lat,
        'radius_lon': radius_lon
    }

    # Visualize covering circles
    GeoUtils.plot_circles(circles, df, dense_centers)

    return circles


# Add visualization method to GeoUtils
GeoUtils.plot_circles = staticmethod(lambda circles, df, dense_centers=None:
                                     (plt.figure(figsize=(12, 10)),
                                      plt.scatter(df['Longitude'], df['Latitude'], c='lightblue', alpha=0.5, s=10,
                                                  label='Data Points'),
                                      [plt.gca().add_patch(Circle((lon, lat), circles['radius_lon'],
                                                                  fill=False, edgecolor='red', linewidth=1, alpha=0.7))
                                       for lat, lon in circles['centers']],
                                      [plt.text(lon, lat, f"{i}", fontsize=8) for i, (lat, lon) in
                                       enumerate(circles['centers'])],
                                      plt.scatter([p[1] for p in dense_centers],
                                                  [p[0] for p in dense_centers] if dense_centers else [],
                                                  c='green', s=100, alpha=0.6, marker='*',
                                                  label='Dense Region Centers') if dense_centers else None,
                                      plt.xlabel('Longitude'), plt.ylabel('Latitude'),
                                      plt.title('Balanced Equal-Area Covering Circles & Data Distribution (Multi-Day)'),
                                      plt.legend(), plt.tight_layout(),
                                      plt.savefig('multi_day_balanced_circle_coverage.png', dpi=300), plt.close())
                                     )


# Force balanced data point assignment
def assign_points_to_circles(df, circles, max_points_per_region=None):
    """Force balanced data point assignment across regions"""
    centers = circles['centers']
    n_regions = len(centers)
    total_points = len(df)

    if max_points_per_region is None:
        avg_points = total_points / n_regions
        max_points_per_region = int(avg_points * 1.2)
        print(f"Automatically set max points per region: {max_points_per_region}")

    df['region_id'] = -1
    df['distance_to_center'] = float('inf')

    # Calculate distances from all points to all circle centers
    distances = []
    for idx, row in df.iterrows():
        point_distances = []
        for i, (lat, lon) in enumerate(centers):
            dist = GeoUtils.haversine_distance(row['Latitude'], row['Longitude'], lat, lon)
            point_distances.append((i, dist))
        distances.append(sorted(point_distances, key=lambda x: x[1]))

    # Initialize region counters
    region_counts = {i: 0 for i in range(n_regions)}
    unassigned = list(range(total_points))

    # First pass: Assign to nearest region if not full
    for idx in range(total_points):
        if region_counts[distances[idx][0][0]] < max_points_per_region:
            region_id, dist = distances[idx][0]
            df.at[idx, 'region_id'] = region_id
            df.at[idx, 'distance_to_center'] = dist
            region_counts[region_id] += 1
            unassigned.remove(idx)

    # Second pass: Handle unassigned points
    print(f"Unassigned points after first pass: {len(unassigned)}")
    for idx in unassigned:
        assigned = False
        # Try top 10 nearest regions
        for i in range(min(10, len(distances[idx]))):
            region_id, dist = distances[idx][i]
            if region_counts[region_id] < max_points_per_region:
                df.at[idx, 'region_id'] = region_id
                df.at[idx, 'distance_to_center'] = dist
                region_counts[region_id] += 1
                assigned = True
                break

        # If still unassigned, assign to least populated region
        if not assigned:
            min_count = min(region_counts.values())
            candidates = [rid for rid, cnt in region_counts.items() if cnt == min_count]
            candidate_distances = [(rid, GeoUtils.haversine_distance(
                df.at[idx, 'Latitude'], df.at[idx, 'Longitude'],
                centers[rid][0], centers[rid][1])) for rid in candidates]
            region_id = min(candidate_distances, key=lambda x: x[1])[0]

            df.at[idx, 'region_id'] = region_id
            df.at[idx, 'distance_to_center'] = candidate_distances[0][1]
            region_counts[region_id] += 1

    # Display assignment statistics
    region_counts = df['region_id'].value_counts().sort_index()
    print("\nBalanced region point distribution:")
    print(f"Mean: {region_counts.mean():.1f}")
    print(f"Standard Deviation: {region_counts.std():.1f}")
    print(f"Max: {region_counts.max()}")
    print(f"Min: {region_counts.min()}")

    # Display daily region distribution
    daily_distribution = pd.crosstab(df['region_id'], df['Date'])
    print("\nDaily point distribution per region:")
    print(daily_distribution)

    return df


# Build adjacency matrices
def build_geographic_matrix(circles):
    """Build geographic adjacency matrix"""
    centers = circles['centers']
    n = len(centers)
    matrix = np.zeros((n, n))

    all_distances = []
    for i in range(n):
        lat1, lon1 = centers[i]
        for j in range(i + 1, n):
            lat2, lon2 = centers[j]
            dist = GeoUtils.haversine_distance(lat1, lon1, lat2, lon2)
            all_distances.append(dist)

    sigma = np.mean(all_distances) / 2 if all_distances else circles['radius_m'] * 2

    for i in range(n):
        lat1, lon1 = centers[i]
        for j in range(n):
            if i != j:
                lat2, lon2 = centers[j]
                distance = GeoUtils.haversine_distance(lat1, lon1, lat2, lon2)
                matrix[i][j] = np.exp(-(distance ** 2) / (2 * sigma ** 2))

    return np.clip(matrix, 0, None)


def build_flow_matrix(df):
    """Build order flow matrix (multi-day average)"""
    n = 30
    matrix = np.zeros((n, n))
    total_orders = df.shape[0]
    if total_orders == 0:
        return matrix

    # Calculate flow matrix for each day and average
    for date in df['Date'].unique():
        date_df = df[df['Date'] == date]

        for i in range(n):
            region_i = date_df[date_df['region_id'] == i]
            count_i = len(region_i)
            if count_i == 0:
                continue

            for j in range(n):
                if i != j:
                    region_j = date_df[date_df['region_id'] == j]
                    count_j = len(region_j)
                    if count_j == 0:
                        continue

                    matrix[i][j] += (count_i * count_j) / total_orders

    # Average over days
    matrix = matrix / len(df['Date'].unique())

    # Safe normalization
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        normalized = matrix / row_sums
    normalized[np.isnan(normalized)] = 0

    return np.clip(normalized, 0, None)


def build_functional_matrix(df):
    """Build functional similarity matrix (multi-day average)"""
    daily_matrices = []

    for date in df['Date'].unique():
        date_df = df[df['Date'] == date]
        region_category = pd.crosstab(date_df['region_id'], date_df['Food_Category'])

        # Fill missing regions with 0
        for i in range(30):
            if i not in region_category.index:
                region_category.loc[i] = 0
        region_category = region_category.sort_index()

        daily_matrices.append(cosine_similarity(region_category))

    # Average over days
    func_matrix = np.mean(daily_matrices, axis=0)
    return np.clip(func_matrix, 0, None)


def visualize_matrix(matrix, title, save_path=None):
    """Visualize adjacency matrix as heatmap"""
    # Clip negative values if exist
    if np.any(matrix < 0):
        print(f"Warning: Negative values found in {title}, automatically clipped")
        matrix = np.clip(matrix, 0, None)

    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(matrix)
    np.fill_diagonal(mask, 1)  # Hide diagonal elements

    sns.heatmap(matrix, mask=mask, cmap='YlOrRd',
                annot=False, square=True, linewidths=.5,
                cbar_kws={"label": "Weight Value"})

    plt.title(title, fontsize=15)
    plt.xticks(ticks=np.arange(30) + 0.5, labels=[f"Region {i}" for i in range(30)],
               rotation=90, fontsize=6)
    plt.yticks(ticks=np.arange(30) + 0.5, labels=[f"Region {i}" for i in range(30)],
               rotation=0, fontsize=6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Main function
def main(file_paths, output_dir='./', n_regions=30):
    """Main pipeline to generate balanced adjacency matrices"""
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load and merge multi-day data
        df = load_multiple_days_data(file_paths)
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        return None, None, None

    try:
        # Generate balanced equal-area circles
        print(f"Generating {n_regions} balanced equal-area covering circles...")
        circles = create_balanced_equal_area_circles(df, n_circles=n_regions)

        # Force balanced point assignment
        df = assign_points_to_circles(df, circles)

        # Save circle information
        with open(f"{output_dir}/multi_day_balanced_circle_info.txt", 'w', encoding='utf-8') as f:
            f.write(f"Number of circles: {n_regions}\n")
            f.write(f"Each circle area: {np.pi * circles['radius_m'] ** 2:.2f} square meters\n")
            f.write(f"Each circle radius: {circles['radius_m']:.2f} meters\n")
            f.write(f"Date range: {sorted(df['Date'].unique())}\n")
            f.write("Circle center coordinates (Latitude, Longitude):\n")
            for i, (lat, lon) in enumerate(circles['centers']):
                f.write(f"Region {i}: {lat:.6f}, {lon:.6f}\n")

        # Save point-region mapping
        df[['ID', 'Date', 'Latitude', 'Longitude', 'region_id', 'distance_to_center']].to_csv(
            f"{output_dir}/multi_day_region_mapping.csv", index=False)
    except Exception as e:
        print(f"Region division failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

    try:
        # Build adjacency matrices
        print("Building geographic adjacency matrix...")
        geo_matrix = build_geographic_matrix(circles)

        print("Building order flow matrix...")
        flow_matrix = build_flow_matrix(df)

        print("Building functional similarity matrix...")
        func_matrix = build_functional_matrix(df)
    except Exception as e:
        print(f"Matrix construction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

    # Validate matrix dimensions
    try:
        assert geo_matrix.shape == (n_regions, n_regions), f"Geographic matrix dimension error: {geo_matrix.shape}"
        assert flow_matrix.shape == (n_regions, n_regions), f"Flow matrix dimension error: {flow_matrix.shape}"
        assert func_matrix.shape == (n_regions, n_regions), f"Functional matrix dimension error: {func_matrix.shape}"
    except AssertionError as e:
        print(f"Matrix validation failed: {str(e)}")
        return None, None, None

    try:
        # Save matrices (compatible with MGCN-STF model)
        np.savetxt(f"{output_dir}/geographic_adjacency.csv", geo_matrix, delimiter=',')
        np.savetxt(f"{output_dir}/order_flow.csv", flow_matrix, delimiter=',')
        np.savetxt(f"{output_dir}/functional_similarity.csv", func_matrix, delimiter=',')

        # Visualize matrices
        visualize_matrix(geo_matrix, "Geographic Adjacency Matrix ",
                         f"{output_dir}/geographic_adjacency.png")
        visualize_matrix(flow_matrix, "Order Flow Matrix ",
                         f"{output_dir}/order_flow.png")
        visualize_matrix(func_matrix, "Functional Similarity Matrix ",
                         f"{output_dir}/functional_similarity.png")

        print(f"matrices generated successfully! Saved to: {os.path.abspath(output_dir)}")
        return geo_matrix, flow_matrix, func_matrix
    except Exception as e:
        print(f"File saving failed: {str(e)}")
        return None, None, None


if __name__ == "__main__":
    # Placeholder data file paths (replace with your actual paths)
    data_files = [
        'path/to/your/data/Delivery_Orders.xlsx',
      '  ........'
    ]

    # Check for missing files
    missing_files = []
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        print("Error: The following files are missing:")
        for file in missing_files:
            print(f"- {file}")
        print("\nPlease verify file paths and names")
    else:
        geo_mat, flow_mat, func_mat = main(
            data_files,
            output_dir="./multi_day_balanced_matrix_results",
            n_regions=30
        )