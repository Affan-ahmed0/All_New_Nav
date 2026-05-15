import streamlit as st
import pandas as pd
import json
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.validation import make_valid
from shapely.ops import unary_union
import zipfile
import io

# --- 1. UI CONFIGURATION ---
st.set_page_config(page_title="City in Squares", layout="wide")
st.title("🏙️ Ur City in Squares")

with st.sidebar:
    st.header("Aesthetics")
    primary_color = st.color_picker("Accent Color", "#E25C5C")
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload Timeline Data", type=["zip", "json"])
    st.header("Geography")
    target_city = st.text_input("City Name", value="Manhattan, New York")
    grid_size = st.slider("Grid Density", 10, 60, 35)

# --- 2. GEOMETRY ENGINE ---
def simplify_vignelli(poly):
    """Smoothes and snaps to 45/90 degree angles while maintaining valid geometry."""
    if isinstance(poly, MultiPolygon):
        parts = [simplify_vignelli(p) for p in poly.geoms]
        return make_valid(unary_union(parts))
    
    # Handle single Polygon
    poly = poly.simplify(0.001, preserve_topology=True)
    coords = list(poly.exterior.coords)
    new_coords = [coords[0]]
    
    for i in range(1, len(coords)):
        prev = new_coords[-1]
        curr = coords[i]
        dx, dy = curr[0] - prev[0], curr[1] - prev[1]
        if abs(dx) < 0.00001 and abs(dy) < 0.00001: continue
        
        angle = np.arctan2(dy, dx)
        snapped_angle = round(angle / (np.pi / 4)) * (np.pi / 4)
        dist = np.sqrt(dx**2 + dy**2)
        new_coords.append((prev[0] + dist * np.cos(snapped_angle), 
                           prev[1] + dist * np.sin(snapped_angle)))
    
    if new_coords[0] != new_coords[-1]:
        new_coords.append(new_coords[0])
    
    v_poly = Polygon(new_coords)
    return make_valid(v_poly.buffer(0.0001).buffer(-0.0001))

def process_timeline(file):
    if file.name.endswith('.zip'):
        with zipfile.ZipFile(file) as z:
            json_files = [f for f in z.namelist() if f.endswith('.json')]
            with z.open(json_files[0]) as f:
                data = json.load(f)
    else:
        data = json.load(file)
    
    points = []
    if "timelineEdits" in data:
        for edit in data["timelineEdits"]:
            aggregates = edit.get("placeAggregates", {}).get("placeAggregateInfo", [])
            for info in aggregates:
                pt = info.get("point", {})
                lat, lon = pt.get("latE7"), pt.get("lngE7")
                if lat and lon: points.append([lat/1e7, lon/1e7])
    else:
        locs = data.get('locations', data.get('records', []))
        for entry in locs:
            lat = entry.get('latitudeE7') or entry.get('latitude_e7')
            lon = entry.get('longitudeE7') or entry.get('longitude_e7')
            if lat and lon: points.append([lat/1e7, lon/1e7])
                
    return pd.DataFrame(points, columns=['lat', 'lon'])

# --- 3. MAIN EXECUTION ---
if uploaded_file and target_city:
    status = st.status("Crafting Sticker...")
    
    try:
        # Improved Geocoding: Specifically filter for administrative polygons
        city_gdf = ox.geocode_to_gdf(target_city, which_result=None)
        
        # Filter only Polygon and MultiPolygon entries
        shapes = city_gdf[city_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        
        if shapes.empty:
            st.error(f"Could not find a boundary for '{target_city}'. Try adding a state or country (e.g. 'Mumbai, Maharashtra').")
            st.stop()
            
        raw_geometry = unary_union(shapes.geometry.tolist())
        vignelli_boundary = simplify_vignelli(raw_geometry)
    except Exception as e:
        st.error(f"Geocoding Error: {e}")
        st.stop()

    df_pts = process_timeline(uploaded_file)
    xmin, ymin, xmax, ymax = vignelli_boundary.bounds
    width, height = (xmax - xmin) / grid_size, (ymax - ymin) / grid_size
    
    fig, ax = plt.subplots(figsize=(10, 12), dpi=100)
    ax.set_axis_off()
    ax.set_xlim(xmin - width, xmax + width)
    ax.set_ylim(ymin - height, ymax + height)
    
    # 1. THE STICKER BODY
    bound_parts = [vignelli_boundary] if not hasattr(vignelli_boundary, 'geoms') else vignelli_boundary.geoms
    for b in bound_parts:
        if hasattr(b, 'exterior'):
            ax.add_patch(plt.Polygon(np.array(b.exterior.coords), facecolor='white', 
                                    edgecolor='#F0F0F0', lw=12, zorder=1, alpha=0.9))

    # 2. THE TILES
    for i in range(grid_size):
        for j in range(grid_size):
            sq_x, sq_y = xmin + i * width, ymin + j * height
            sq_box = box(sq_x, sq_y, sq_x + width, sq_y + height)
            
            try:
                if not vignelli_boundary.intersects(sq_box): continue
                clipped_tile = sq_box.intersection(vignelli_boundary)
                if clipped_tile.is_empty: continue
            except: continue
            
            mask = (df_pts['lon'] >= sq_x) & (df_pts['lon'] < sq_x + width) & \
                   (df_pts['lat'] >= sq_y) & (df_pts['lat'] < sq_y + height)
            count = mask.sum()
            
            parts = clipped_tile.geoms if hasattr(clipped_tile, 'geoms') else [clipped_tile]
            for part in parts:
                if hasattr(part, 'exterior'):
                    coords = np.array(part.exterior.coords)
                    if count > 0:
                        h_density = min(max(1, int(count / 3)), 6)
                        ax.add_patch(plt.Polygon(coords, facecolor='none', 
                                                hatch='/' * h_density, edgecolor=primary_color, 
                                                lw=0.4, zorder=5))
                    else:
                        ax.add_patch(plt.Polygon(coords, facecolor='none', 
                                                edgecolor=primary_color, lw=0.15, alpha=0.2, zorder=2))

    # 3. THIN BOUNDARY & TEXT
    for b in bound_parts:
        if hasattr(b, 'exterior'):
            ax.add_patch(plt.Polygon(np.array(b.exterior.coords), facecolor='none', 
                                    edgecolor=primary_color, lw=1.2, zorder=10))
    
    plt.text(0.5, 0.02, "ur city in squares", transform=ax.transAxes, 
             ha='center', fontsize=24, color='#333333', fontweight='light')

    status.update(label="Sticker Ready!", state="complete", expanded=False)
    fig.patch.set_facecolor('none')
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches='tight', pad_inches=0.3)
    st.download_button("Download Sticker", buf.getvalue(), "vignelli_sticker.png", "image/png")