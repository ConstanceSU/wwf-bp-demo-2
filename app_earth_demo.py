import ast
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from questionnaire import render_questionnaire


DEV_MODE = False


st.set_page_config(
    page_title="Nature Intelligence Platform | Earth Demo",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if not DEV_MODE:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"],
            [data-testid="collapsedControl"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            max-width: none;
        }
        .main-title {
            font-size: 2.15rem;
            font-weight: 750;
            margin-bottom: 0.35rem;
            color: #0f172a;
        }
        .sub-title {
            font-size: 1.02rem;
            color: #4f5f6f;
            margin-bottom: 1.4rem;
        }
        .questionnaire-anchor {
            border-top: 1px solid #e6edf1;
            margin-top: 0rem;
            padding-top: 1.3rem;
            background: #ffffff;
            max-width: 1240px;
            margin-left: auto;
            margin-right: auto;
        }
        .card {
            padding: 1rem 1rem 0.85rem 1rem;
            border-radius: 8px;
            border: 1px solid #e7e9ee;
            background-color: #ffffff;
            box-shadow: 0 4px 14px rgba(15,23,42,0.05);
            margin-bottom: 1rem;
        }
        .dashboard-card {
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e7e9ee;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(15,23,42,0.05);
            margin-bottom: 1rem;
        }
        .dashboard-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .input-row {
            border-bottom: 1px solid #f0f2f5;
            padding: 0.48rem 0;
        }
        .input-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.08rem;
        }
        .input-value {
            color: #111827;
            font-size: 0.92rem;
            font-weight: 600;
        }
        .badge {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 700;
            margin: 0.15rem 0.2rem 0.15rem 0;
            border: 1px solid transparent;
        }
        .badge-green {
            background: #ecfdf3;
            color: #087443;
            border-color: #bbf7d0;
        }
        .badge-blue {
            background: #eff6ff;
            color: #1d4ed8;
            border-color: #bfdbfe;
        }
        .badge-orange {
            background: #fff7ed;
            color: #c2410c;
            border-color: #fed7aa;
        }
        .metric-box {
            padding: 0.85rem;
            border-radius: 8px;
            background: #f8fafc;
            border: 1px solid #edf2f7;
            min-height: 92px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.8rem;
        }
        .metric-label {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.18rem;
            font-weight: 800;
        }
        .section-note {
            color: #64748b;
            font-size: 0.86rem;
        }
        .score-row {
            margin-bottom: 0.42rem;
        }
        .score-head {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.08rem;
        }
        .score-name {
            font-weight: 800;
            color: #1f2937;
            font-size: 0.88rem;
        }
        .score-points {
            color: #334155;
            font-weight: 700;
            font-size: 0.86rem;
            white-space: nowrap;
        }
        .score-explain {
            color: #6b7280;
            font-size: 0.76rem;
            line-height: 1.18;
            margin-bottom: 0.16rem;
        }
        .score-track {
            height: 5px;
            background: #e8eef7;
            border-radius: 999px;
            overflow: hidden;
        }
        .score-fill {
            height: 100%;
            background: #2f80ed;
            border-radius: 999px;
        }
        .score-divider {
            border-top: 1px solid #e7e9ee;
            margin: 0.75rem 0 0.65rem 0;
        }
        .score-total {
            color: #0f172a;
            font-size: 1.45rem;
            font-weight: 800;
        }
        .option-card {
            display: grid;
            grid-template-columns: 56px 1fr auto;
            gap: 0.9rem;
            align-items: center;
            padding: 0.9rem 1rem;
            border: 1px solid #e1e7ef;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(15,23,42,0.04);
            background: #ffffff;
            margin-bottom: 0.75rem;
        }
        .option-icon {
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 1.45rem;
            font-weight: 800;
        }
        .option-title {
            color: #0f172a;
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.1rem;
        }
        .option-subtitle {
            color: #64748b;
            font-size: 0.9rem;
        }
        .option-score {
            color: #0f172a;
            font-size: 1.4rem;
            font-weight: 800;
            white-space: nowrap;
        }
        div.stButton > button {
            min-height: 48px;
            font-size: 1rem;
            font-weight: 650;
            border-radius: 8px;
        }
        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def html_fragment(markup):
    return "\n".join(line.strip() for line in textwrap.dedent(markup).strip().splitlines())


def query_param_is_true(name):
    try:
        value = st.query_params.get(name, "")
    except AttributeError:
        value = st.experimental_get_query_params().get(name, [""])
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value).lower() in {"1", "true", "yes", "y"}


def load_app_3_reusable_objects():
    """
    Reuse app_3's functions and constants without importing app_3 directly.
    Importing app_3 would execute its Streamlit UI, so this keeps the existing
    working demo untouched while sharing the same recommendation flow.
    """
    source_path = Path("app_3.py")
    source = source_path.read_text()
    module = ast.parse(source, filename=str(source_path))
    reusable_nodes = []

    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            reusable_nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROJECT_COLUMNS":
                    reusable_nodes.append(node)

    reusable_module = ast.Module(body=reusable_nodes, type_ignores=[])
    ast.fix_missing_locations(reusable_module)

    namespace = {
        "ast": ast,
        "html": __import__("html"),
        "textwrap": textwrap,
        "Path": Path,
        "np": np,
        "pd": pd,
        "st": st,
        "render_questionnaire": render_questionnaire,
    }
    exec(compile(reusable_module, str(source_path), "exec"), namespace)

    app_2_helpers = namespace["load_app_2_functions"]()
    for name in [
        "build_recommendation",
        "ensure_columns",
        "find_first_existing",
        "format_currency",
        "get_financial_view",
    ]:
        namespace[name] = app_2_helpers[name]

    return namespace


APP_3 = load_app_3_reusable_objects()

find_first_existing = APP_3["find_first_existing"]
ensure_columns = APP_3["ensure_columns"]
PROJECT_COLUMNS = APP_3["PROJECT_COLUMNS"]
build_questionnaire_recommendation = APP_3["build_questionnaire_recommendation"]
render_recommendation_outputs = APP_3["render_recommendation_outputs"]


@st.cache_data
def load_data_for_earth_demo():
    base = Path(".")
    company_path = base / "company_exposure.csv"
    project_path = base / "nbs_projects_matching_ready.csv"
    finance_path = base / "project_financial_output.csv"

    if not company_path.exists():
        return None, None, None, "Missing company_exposure.csv."
    if not project_path.exists():
        return None, None, None, "Missing nbs_projects_matching_ready.csv."

    df_company = pd.read_csv(company_path)
    df_projects = pd.read_csv(project_path)
    df_finance = pd.read_csv(finance_path) if finance_path.exists() else pd.DataFrame()
    data_note = "Using nbs_projects_matching_ready.csv as the main project supply database."

    df_company = ensure_columns(df_company, [
        "company_name",
        "hotspots",
        "risk_exposure",
        "sbtn_targets",
        "business_priorities",
        "company_summary",
    ])

    df_projects = ensure_columns(df_projects, PROJECT_COLUMNS)

    if not df_finance.empty:
        df_finance = ensure_columns(df_finance, [
            "project_id",
            "project_name",
            "country",
            "intervention_type",
            "estimated_cost_usd",
        ])

    return df_company, df_projects, df_finance, data_note


COUNTRY_CENTROIDS = {
    "Australia": (-25.2744, 133.7751),
    "Austria": (47.5162, 14.5501),
    "Belize": (17.1899, -88.4976),
    "Brazil": (-14.2350, -51.9253),
    "Cameroon": (7.3697, 12.3547),
    "Canada": (56.1304, -106.3468),
    "Chile": (-35.6751, -71.5430),
    "Colombia": (4.5709, -74.2973),
    "Ecuador": (-1.8312, -78.1834),
    "Finland": (61.9241, 25.7482),
    "Greece": (39.0742, 21.8243),
    "Guatemala": (15.7835, -90.2308),
    "Honduras": (15.2000, -86.2419),
    "Hungary": (47.1625, 19.5033),
    "India": (20.5937, 78.9629),
    "Italy": (41.8719, 12.5674),
    "Laos": (19.8563, 102.4955),
    "Madagascar": (-18.7669, 46.8691),
    "Malaysia": (4.2105, 101.9758),
    "Mexico": (23.6345, -102.5528),
    "Myanmar": (21.9162, 95.9560),
    "Nepal": (28.3949, 84.1240),
    "Pakistan": (30.3753, 69.3451),
    "Paraguay": (-23.4425, -58.4438),
    "Peru": (-9.1900, -75.0152),
    "Philippines": (12.8797, 121.7740),
    "Portugal": (39.3999, -8.2245),
    "Romania": (45.9432, 24.9668),
    "Slovakia": (48.6690, 19.6990),
    "South Africa": (-30.5595, 22.9375),
    "Tanzania": (-6.3690, 34.8888),
    "Thailand": (15.8700, 100.9925),
    "Tunisia": (33.8869, 9.5375),
    "Uganda": (1.3733, 32.2903),
    "United Arab Emirates": (23.4241, 53.8478),
    "United Kingdom": (55.3781, -3.4360),
    "UK": (55.3781, -3.4360),
    "USA": (37.0902, -95.7129),
    "Vietnam": (14.0583, 108.2772),
    "Zimbabwe": (-19.0154, 29.1549),
}


def clean_marker_value(value, default="To be assessed"):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return str(value).strip()


def truncate_text(value, max_length=92):
    value = clean_marker_value(value)
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "…"


def compact_tag_list(value, max_items=3):
    value = clean_marker_value(value)
    if value == "To be assessed":
        return value
    tags = [part.strip() for part in value.replace("|", ";").split(";") if part.strip()]
    if not tags:
        return value
    visible = tags[:max_items]
    suffix = f" +{len(tags) - max_items} more" if len(tags) > max_items else ""
    return "; ".join(visible) + suffix


def split_location_names(location):
    if pd.isna(location) or str(location).strip() == "":
        return []
    return [
        part.strip()
        for part in str(location).replace(" and ", ",").split(",")
        if part.strip()
    ]


def centroid_for_location(location):
    names = split_location_names(location)
    points = [COUNTRY_CENTROIDS[name] for name in names if name in COUNTRY_CENTROIDS]
    if not points:
        return None, "unmapped"
    lat = sum(point[0] for point in points) / len(points)
    lon = sum(point[1] for point in points) / len(points)
    return (lat, lon), "approximate country centroid" if len(points) == 1 else "approximate multi-country centroid"


@st.cache_data
def load_nbs_project_locations():
    """
    Load the real project supply dataset used by the recommendation engine.
    The current file has country-level geography but no lat/lon, so coordinates
    are resolved later to approximate country or multi-country centroids.
    """
    project_path = Path("nbs_projects_matching_ready.csv")
    if not project_path.exists():
        return pd.DataFrame()
    return pd.read_csv(project_path)


def build_project_marker_data(df_projects):
    """
    Build marker objects for the globe.

    Exact latitude/longitude columns are used if they exist. The current demo
    dataset does not include them, so markers are mapped to country centroids.
    Repeated country markers receive a small deterministic spiral offset so all
    projects remain visible instead of sitting directly on top of each other.
    """
    if df_projects is None or df_projects.empty:
        return []

    lat_col = find_first_existing(df_projects, ["latitude", "lat", "project_latitude"])
    lon_col = find_first_existing(df_projects, ["longitude", "lon", "lng", "project_longitude"])
    markers = []
    location_counts = {}

    for index, row in df_projects.reset_index(drop=True).iterrows():
        location = clean_marker_value(row.get("country"), "")
        coordinate_source = "exact"

        lat = pd.to_numeric(row.get(lat_col), errors="coerce") if lat_col else np.nan
        lon = pd.to_numeric(row.get(lon_col), errors="coerce") if lon_col else np.nan

        if pd.isna(lat) or pd.isna(lon):
            centroid, coordinate_source = centroid_for_location(location)
            if centroid is None:
                continue
            lat, lon = centroid

        key = location or f"{round(float(lat), 2)},{round(float(lon), 2)}"
        duplicate_index = location_counts.get(key, 0)
        location_counts[key] = duplicate_index + 1

        if duplicate_index:
            angle = duplicate_index * 2.399963
            radius = 0.45 + 0.12 * (duplicate_index // 7)
            lat = float(lat) + np.sin(angle) * radius
            lon = float(lon) + np.cos(angle) * radius
            coordinate_source = f"{coordinate_source}, offset for overlapping projects"

        markers.append({
            "id": clean_marker_value(row.get("project_id"), f"project-{index + 1}"),
            "name": truncate_text(row.get("project_name"), 76),
            "location": location or clean_marker_value(row.get("continent"), "Location to be assessed"),
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "pressure": compact_tag_list(row.get("main_pressure_tags"), 3),
            "intervention": compact_tag_list(row.get("intervention_type_tags"), 3),
            "coordinate_source": coordinate_source,
        })

    return markers


def render_unified_earth_hero(project_markers):
    marker_payload = json.dumps(project_markers, ensure_ascii=False).replace("</", "<\\/")
    globe_html = """
        <div id="earth-root">
            <canvas id="earth-canvas" aria-label="Interactive nature risk globe"></canvas>
            <div class="hero-shade"></div>
            <section class="hero-copy" aria-label="WWF Nature Intelligence Platform">
                <div class="hero-eyebrow">WWF NATURE INTELLIGENCE PLATFORM</div>
                <h1>From nature pressure to investable action.</h1>
                <p class="hero-subtitle">
                    Explore priority geographies, understand the pressure on nature, and move from
                    business context to credible nature-based solutions opportunities.
                </p>
            </section>
            <div class="info-panel" id="info-panel">
                <div class="info-title" id="info-title">NbS project portfolio</div>
                <div class="info-location" id="info-location">Hover over a marker to inspect a project.</div>
                <div class="info-line"><span>Nature pressure</span><strong id="info-pressure">Project pressure drivers will appear here.</strong></div>
                <div class="info-line"><span>Intervention</span><strong id="info-nbs">Nature-based solution type will appear here.</strong></div>
            </div>
            <div class="fallback" id="fallback">
                Interactive globe unavailable. The questionnaire and recommendation engine below still work normally.
            </div>
        </div>
        <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
        <script>
        (function () {
            const projectMarkers = __PROJECT_MARKERS__;
            const fallback = document.getElementById("fallback");
            if (!window.THREE) {
                fallback.style.display = "block";
                return;
            }

            const root = document.getElementById("earth-root");
            const canvas = document.getElementById("earth-canvas");
            const panel = document.getElementById("info-panel");
            const title = document.getElementById("info-title");
            const locationText = document.getElementById("info-location");
            const pressure = document.getElementById("info-pressure");
            const nbs = document.getElementById("info-nbs");
            const AUTO_ROTATION_SPEED = 0.00115;

            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(42, root.clientWidth / root.clientHeight, 0.1, 100);
            camera.position.z = 3.85;

            const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.setSize(root.clientWidth, root.clientHeight);

            const group = new THREE.Group();
            scene.add(group);

            function layoutGlobe() {
                const isCompact = root.clientWidth < 760;
                group.position.x = isCompact ? 0 : 0.78;
                group.position.y = isCompact ? -0.56 : 0.08;
                group.scale.setScalar(isCompact ? 0.88 : 1.08);
            }

            const textureLoader = new THREE.TextureLoader();
            textureLoader.crossOrigin = "anonymous";
            const earthTexture = textureLoader.load(
                "https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg",
                undefined,
                undefined,
                () => { fallback.style.display = "block"; }
            );
            const bumpTexture = textureLoader.load(
                "https://unpkg.com/three-globe/example/img/earth-topology.png"
            );

            const globe = new THREE.Mesh(
                new THREE.SphereGeometry(1.16, 128, 128),
                new THREE.MeshPhongMaterial({
                    map: earthTexture,
                    bumpMap: bumpTexture,
                    bumpScale: 0.035,
                    specular: 0x244a55,
                    shininess: 11
                })
            );
            group.add(globe);

            const atmosphere = new THREE.Mesh(
                new THREE.SphereGeometry(1.235, 96, 96),
                new THREE.MeshBasicMaterial({
                    color: 0x8fd7ff,
                    transparent: true,
                    opacity: 0.14,
                    side: THREE.BackSide
                })
            );
            group.add(atmosphere);

            scene.add(new THREE.AmbientLight(0xbfd8ff, 0.74));
            const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
            keyLight.position.set(2.2, 1.4, 3.2);
            scene.add(keyLight);
            const rimLight = new THREE.DirectionalLight(0x62d5ff, 0.82);
            rimLight.position.set(-2.4, -1.2, -1);
            scene.add(rimLight);

            const markerGroup = new THREE.Group();
            group.add(markerGroup);
            const markerObjects = [];

            function latLonToVector(lat, lon, radius) {
                const phi = (90 - lat) * Math.PI / 180;
                const theta = (lon + 180) * Math.PI / 180;
                return new THREE.Vector3(
                    -radius * Math.sin(phi) * Math.cos(theta),
                    radius * Math.cos(phi),
                    radius * Math.sin(phi) * Math.sin(theta)
                );
            }

            projectMarkers.forEach((spot) => {
                const position = latLonToVector(spot.lat, spot.lon, 1.215);
                const marker = new THREE.Mesh(
                    new THREE.SphereGeometry(0.025, 20, 20),
                    new THREE.MeshBasicMaterial({ color: 0xffd166 })
                );
                marker.position.copy(position);
                marker.userData = spot;

                const glow = new THREE.Mesh(
                    new THREE.SphereGeometry(0.057, 20, 20),
                    new THREE.MeshBasicMaterial({
                        color: 0x9ff7ca,
                        transparent: true,
                        opacity: 0.28
                    })
                );
                glow.position.copy(position);
                glow.userData = spot;

                markerGroup.add(glow);
                markerGroup.add(marker);
                markerObjects.push(marker, glow);
            });

            function setInfo(spot) {
                title.textContent = spot.name;
                locationText.textContent = spot.location;
                pressure.textContent = spot.pressure;
                nbs.textContent = spot.intervention;
            }
            if (projectMarkers.length) {
                setInfo(projectMarkers[0]);
            }

            const raycaster = new THREE.Raycaster();
            const pointer = new THREE.Vector2();
            let isDragging = false;
            let lastX = 0;
            let lastY = 0;
            let velocityX = AUTO_ROTATION_SPEED;
            let velocityY = 0;

            function updatePointer(event) {
                const rect = canvas.getBoundingClientRect();
                pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            }

            canvas.addEventListener("pointerdown", (event) => {
                isDragging = true;
                lastX = event.clientX;
                lastY = event.clientY;
                canvas.setPointerCapture(event.pointerId);
            });

            canvas.addEventListener("pointermove", (event) => {
                if (isDragging) {
                    const dx = event.clientX - lastX;
                    const dy = event.clientY - lastY;
                    group.rotation.y += dx * 0.006;
                    group.rotation.x += dy * 0.004;
                    group.rotation.x = Math.max(-0.72, Math.min(0.72, group.rotation.x));
                    velocityX = dx * 0.00035;
                    velocityY = dy * 0.00022;
                    lastX = event.clientX;
                    lastY = event.clientY;
                    return;
                }
                updatePointer(event);
                raycaster.setFromCamera(pointer, camera);
                const hits = raycaster.intersectObjects(markerObjects, false);
                if (hits.length) {
                    setInfo(hits[0].object.userData);
                    panel.classList.add("active");
                    canvas.style.cursor = "pointer";
                } else {
                    canvas.style.cursor = "grab";
                }
            });

            canvas.addEventListener("pointerup", () => {
                isDragging = false;
            });

            canvas.addEventListener("click", (event) => {
                updatePointer(event);
                raycaster.setFromCamera(pointer, camera);
                const hits = raycaster.intersectObjects(markerObjects, false);
                if (hits.length) {
                    setInfo(hits[0].object.userData);
                    panel.classList.add("active");
                }
            });

            function onResize() {
                const width = root.clientWidth;
                const height = root.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
                layoutGlobe();
            }
            window.addEventListener("resize", onResize);
            layoutGlobe();

            const starGeometry = new THREE.BufferGeometry();
            const starVertices = [];
            for (let i = 0; i < 320; i += 1) {
                starVertices.push((Math.random() - 0.5) * 8, (Math.random() - 0.5) * 5, -Math.random() * 4 - 0.6);
            }
            starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starVertices, 3));
            const stars = new THREE.Points(
                starGeometry,
                new THREE.PointsMaterial({ color: 0xd7fff0, size: 0.011, transparent: true, opacity: 0.55 })
            );
            scene.add(stars);

            let tick = 0;
            function animate() {
                requestAnimationFrame(animate);
                tick += 0.01;
                if (!isDragging) {
                    group.rotation.y += velocityX || AUTO_ROTATION_SPEED;
                    group.rotation.x += velocityY;
                    velocityX = velocityX * 0.99 + AUTO_ROTATION_SPEED * 0.01;
                    velocityY *= 0.94;
                }
                markerGroup.children.forEach((child, index) => {
                    if (index % 2 === 0) {
                        child.scale.setScalar(1 + Math.sin(tick * 3 + index) * 0.12);
                    }
                });
                renderer.render(scene, camera);
            }
            fallback.style.display = "none";
            animate();
        }());
        </script>
        <style>
            html, body {
                margin: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                background: #06130f;
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            #earth-root {
                position: relative;
                width: 100%;
                height: 100vh;
                min-height: 760px;
                box-sizing: border-box;
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid rgba(183, 247, 215, 0.13);
                box-shadow: 0 22px 60px rgba(0, 0, 0, 0.24);
                background:
                    radial-gradient(circle at 75% 32%, rgba(130, 240, 188, 0.18), transparent 30%),
                    radial-gradient(circle at 24% 77%, rgba(20, 184, 166, 0.12), transparent 28%),
                    linear-gradient(145deg, #061d15 0%, #062016 44%, #07120f 100%);
            }
            #earth-canvas {
                position: absolute;
                inset: 0;
                z-index: 1;
                width: 100%;
                height: 100%;
                display: block;
                cursor: grab;
            }
            .hero-shade {
                position: absolute;
                inset: 0;
                z-index: 2;
                pointer-events: none;
                background:
                    linear-gradient(90deg, rgba(3, 20, 14, 0.96) 0%, rgba(3, 20, 14, 0.84) 34%, rgba(3, 20, 14, 0.28) 58%, rgba(3, 20, 14, 0.04) 100%),
                    linear-gradient(0deg, rgba(3, 20, 14, 0.25), transparent 42%);
            }
            .hero-copy {
                position: relative;
                z-index: 3;
                width: min(46%, 560px);
                min-width: 430px;
                padding: 62px 0 0 44px;
                color: #eef8f2;
                box-sizing: border-box;
                pointer-events: auto;
            }
            .hero-eyebrow {
                color: #a4f7c8;
                font-size: 12px;
                font-weight: 900;
                letter-spacing: 0.12em;
                margin-bottom: 24px;
            }
            .hero-copy h1 {
                margin: 0;
                max-width: 520px;
                color: #f3faf6;
                font-size: 56px;
                line-height: 1.03;
                font-weight: 850;
                letter-spacing: 0;
            }
            .hero-subtitle {
                max-width: 530px;
                margin: 26px 0 0 0;
                color: rgba(238, 248, 242, 0.84);
                font-size: 18px;
                line-height: 1.56;
                font-weight: 500;
            }
            .info-panel {
                position: absolute;
                z-index: 4;
                right: 34px;
                bottom: 32px;
                width: min(520px, calc(58% - 48px));
                padding: 12px 14px;
                max-height: 190px;
                overflow: hidden;
                border: 1px solid rgba(183, 247, 215, 0.18);
                border-radius: 8px;
                background: rgba(4, 17, 14, 0.38);
                color: #effff7;
                backdrop-filter: blur(10px);
                box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
                text-shadow: 0 1px 8px rgba(0, 0, 0, 0.8);
            }
            .info-title {
                font-size: 17px;
                font-weight: 800;
                margin-bottom: 4px;
                overflow-wrap: anywhere;
                line-height: 1.15;
            }
            .info-location {
                color: #b9d3c9;
                font-size: 12px;
                margin-bottom: 7px;
            }
            .info-line {
                display: grid;
                grid-template-columns: 96px 1fr;
                gap: 7px;
                margin: 4px 0;
                font-size: 12px;
                line-height: 1.28;
            }
            .info-line span {
                color: #a7bdb3;
            }
            .info-line strong {
                color: #f8fffb;
                font-weight: 700;
                overflow-wrap: anywhere;
            }
            .fallback {
                display: none;
                position: absolute;
                z-index: 5;
                inset: 20px;
                color: #e7fff4;
                border: 1px solid rgba(231, 255, 244, 0.22);
                border-radius: 8px;
                padding: 18px;
                background: rgba(6, 19, 15, 0.84);
            }
            @media (max-width: 680px) {
                #earth-root {
                    height: 100vh;
                    min-height: 760px;
                }
                .hero-shade {
                    background:
                        linear-gradient(180deg, rgba(3, 20, 14, 0.98) 0%, rgba(3, 20, 14, 0.92) 38%, rgba(3, 20, 14, 0.28) 68%, rgba(3, 20, 14, 0.08) 100%);
                }
                .hero-copy {
                    width: 100%;
                    min-width: 0;
                    padding: 34px 22px 0 22px;
                }
                .hero-copy h1 {
                    font-size: 40px;
                    max-width: 100%;
                }
                .hero-subtitle {
                    font-size: 15px;
                    margin-top: 18px;
                    max-width: 100%;
                }
                .info-panel {
                    left: 18px;
                    right: 18px;
                    bottom: 18px;
                    width: auto;
                    max-height: 190px;
                }
                .info-line {
                    grid-template-columns: 1fr;
                    gap: 1px;
                }
            }
        </style>
        """
    components.html(
        globe_html
        .replace("__PROJECT_MARKERS__", marker_payload)
        .replace("__PROJECT_COUNT__", str(len(project_markers))),
        height=760,
        scrolling=False,
    )


def render_start_questionnaire_cta():
    st.markdown(
        """
        <style>
            div[data-testid="stButton"] {
                position: relative;
                z-index: 20;
                top: -245px;
                width: 360px;
                height: 0;
                margin-left: 3.7rem;
            }
            div[data-testid="stButton"] > button {
                width: 100%;
                min-height: 54px;
                border-radius: 8px;
                border: 1px solid rgba(164, 247, 200, 0.64);
                background: rgba(164, 247, 200, 0.94);
                color: #062016;
                font-size: 1rem;
                font-weight: 850;
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
            }
            div[data-testid="stButton"] > button:hover {
                border-color: #c2ffd8;
                background: #c2ffd8;
                color: #062016;
            }
            .hero-streamlit-helper {
                position: relative;
                z-index: 20;
                top: -184px;
                width: 360px;
                height: 0;
                margin-left: 3.7rem;
                color: rgba(238, 248, 242, 0.72);
                font-size: 0.88rem;
                line-height: 1.45;
                text-shadow: 0 1px 8px rgba(0, 0, 0, 0.6);
            }
            @media (max-width: 900px) {
                div[data-testid="stButton"] {
                    top: -220px;
                    width: calc(100% - 3rem);
                    margin-left: 1.5rem;
                    margin-right: 1.5rem;
                }
                .hero-streamlit-helper {
                    top: -158px;
                    width: calc(100% - 3rem);
                    margin-left: 1.5rem;
                    margin-right: 1.5rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    start_clicked = st.button(
        "Start company context questionnaire",
        key="start_company_context_questionnaire",
        use_container_width=True,
    )
    st.markdown(
        '<div class="hero-streamlit-helper">Answer a few questions to generate ranked NbS project recommendations.</div>',
        unsafe_allow_html=True,
    )
    return start_clicked


def render_landing_section(project_markers, show_cta):
    render_unified_earth_hero(project_markers)
    if show_cta:
        return render_start_questionnaire_cta()
    return False


def scroll_to_company_context_questionnaire():
    components.html(
        """
        <script>
        window.setTimeout(() => {
            const target = window.parent.document.getElementById("company-context-questionnaire");
            if (target) {
                target.scrollIntoView({ behavior: "smooth", block: "start" });
            }
        }, 250);
        </script>
        """,
        height=0,
    )


df_company, df_projects, df_finance, data_note = load_data_for_earth_demo()

if df_company is None or df_projects is None:
    st.error(
        "The demo data is not available right now. Please check the local setup before running the recommendation."
    )
    st.stop()

df_project_locations = load_nbs_project_locations()
if df_project_locations.empty:
    df_project_locations = df_projects
project_markers = build_project_marker_data(df_project_locations)

company_col = find_first_existing(df_company, ["company_name", "company", "name"])
company_list = sorted(df_company[company_col].dropna().astype(str).unique().tolist())
default_company = "Adidas" if "Adidas" in company_list else company_list[0]
selected_company = default_company
show_top3 = False
show_debug = False

if DEV_MODE:
    st.sidebar.header("Developer settings")
    selected_company = st.sidebar.selectbox(
        "Company record for internal context",
        options=company_list,
        index=company_list.index(default_company),
        key="selected_company_earth_demo",
    )
    show_top3 = st.sidebar.checkbox("Show top 3 projects", value=False, key="show_top3_earth_demo")
    show_debug = st.sidebar.checkbox("Show scoring table", value=False, key="show_debug_earth_demo")
    if show_debug and data_note:
        st.sidebar.caption(f"Developer note: {data_note}")

if "app_3_page" not in st.session_state:
    st.session_state["app_3_page"] = "questionnaire"
if "questionnaire_answers" not in st.session_state:
    st.session_state["questionnaire_answers"] = {}
if "show_questionnaire" not in st.session_state:
    st.session_state["show_questionnaire"] = False
if query_param_is_true("show_questionnaire"):
    st.session_state["show_questionnaire"] = True

if st.session_state["app_3_page"] == "questionnaire":
    start_button_clicked = render_landing_section(
        project_markers,
        show_cta=not st.session_state["show_questionnaire"],
    )
    if start_button_clicked:
        st.session_state["show_questionnaire"] = True
        try:
            st.query_params["show_questionnaire"] = "true"
        except AttributeError:
            st.experimental_set_query_params(show_questionnaire="true")
        st.rerun()
    if st.session_state["show_questionnaire"]:
        st.markdown('<div id="company-context-questionnaire"></div>', unsafe_allow_html=True)
        scroll_to_company_context_questionnaire()
        st.markdown('<div class="questionnaire-anchor">', unsafe_allow_html=True)
        st.markdown("## Company context questionnaire")

        company_profile = render_questionnaire()
        st.markdown("</div>", unsafe_allow_html=True)
        if company_profile is not None:
            st.session_state["questionnaire_answers"] = company_profile.copy()
            st.session_state["company_profile"] = company_profile
            st.session_state["app_3_page"] = "results"
            st.rerun()

else:
    company_profile = st.session_state.get(
        "questionnaire_answers",
        st.session_state.get("company_profile", {}),
    )
    st.session_state["company_profile"] = company_profile

    st.markdown('<div class="main-title">Nature Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">From company context to credible NbS investment options.</div>',
        unsafe_allow_html=True,
    )

    recommendation, error = build_questionnaire_recommendation(
        selected_company,
        company_profile,
        df_projects,
        df_finance,
    )

    if error:
        st.error(
            "We could not generate a recommendation from the available demo data yet. "
            "Your questionnaire answers have been saved."
        )
        if show_debug:
            st.caption(f"Developer note: {error}")
    else:
        st.session_state["recommendation_app_3"] = recommendation
        render_recommendation_outputs(recommendation, company_profile, show_top3, show_debug)
