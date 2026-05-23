import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as ob
import plotly.figure_factory as ff

# Import backend functions
from utils import (
    DEFAULT_TICKERS,
    get_stock_data,
    calculate_returns,
    optimize_classical_mvo,
    generate_efficient_frontier,
    solve_qaoa
)

# Set up Streamlit Page configuration
st.set_page_config(
    page_title="Quantum & Classical Portfolio Optimization",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS for high-end look and glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply font family */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Top Banner Gradient */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 40%, #6d28d9 75%, #db2777 100%);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .main-header h1 {
        font-weight: 800;
        font-size: 2.8rem !important;
        margin: 0;
        letter-spacing: -1.5px;
        text-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-weight: 300;
        font-size: 1.25rem !important;
        opacity: 0.9;
        margin-top: 0.75rem;
        letter-spacing: 0.5px;
    }
    
    /* Custom metric card */
    .custom-card {
        background: rgba(99, 102, 241, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        background: rgba(99, 102, 241, 0.08);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 10px 20px rgba(99, 102, 241, 0.2);
    }
    
    .card-title {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #4f46e5; /* Vibrant indigo for clear visibility */
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .card-value {
        font-size: 2.2rem !important;
        font-weight: 800;
        color: #7c3aed; /* Vibrant violet/purple for high contrast in both themes */
        margin: 0;
    }
    
    .card-desc {
        font-size: 0.8rem !important;
        color: #475569; /* Slate gray for perfect reading contrast */
        margin-top: 0.5rem;
    }
    
    /* Math text container */
    .math-info-box {
        background: rgba(99, 102, 241, 0.07);
        border-left: 4px solid #6366f1;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        color: #cbd5e1;
    }
    
    /* Quantum highlight badges */
    .quantum-badge {
        background: linear-gradient(90deg, #6d28d9 0%, #db2777 100%);
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar adjustments */
    .css-1d391tw {
        background-color: #0f172a;
    }
    
    /* Highlighted table header */
    .table-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Giao diện Sidebar -----------------
st.sidebar.markdown("<div style='margin-bottom: 0px;'><span style='font-size: 50px;'>⚛️</span></div>", unsafe_allow_html=True)
st.sidebar.markdown("<h2 style='font-weight:800; margin-top:0;'>Quantum Portfolio</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 1. Chọn cổ phiếu S&P500
st.sidebar.subheader("📌 1. Chọn Cổ Phiếu")
selected_tickers = st.sidebar.multiselect(
    "Danh sách cổ phiếu đầu tư:",
    options=DEFAULT_TICKERS + ["GOOG", "META", "MS", "GS", "WMT", "KO", "PEP", "PG", "COST", "AMD"],
    default=["AAPL", "MSFT", "NVDA", "JPM", "AMZN"]
)

# Giới hạn số lượng cổ phiếu cho tối ưu hóa lượng tử QAOA
# Đảm bảo số lượng cổ phiếu vừa đủ để mô phỏng lượng tử nhanh chóng (N <= 10)
max_assets = 8
if len(selected_tickers) > max_assets:
    st.sidebar.warning(f"⚠️ Nhằm đảm bảo mô phỏng lượng tử QAOA chạy mượt mà ngay trên CPU, vui lòng chọn tối đa {max_assets} cổ phiếu (Hiện tại: {len(selected_tickers)}).")

# 2. Chọn khoảng thời gian
st.sidebar.subheader("📅 2. Chọn Thời Gian")
today = datetime.date.today()
five_years_ago = today - datetime.timedelta(days=5*365)
start_date = st.sidebar.date_input("Từ ngày:", five_years_ago)
end_date = st.sidebar.date_input("Đến ngày:", today)

# Kiểm tra tính hợp lệ của ngày tháng
if start_date >= end_date:
    st.sidebar.error("Lỗi: Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")

# 3. Chọn mức độ e ngại rủi ro (Risk Aversion Gamma)
st.sidebar.subheader("⚡ 3. Chấp Nhận Rủi Ro")
risk_aversion = st.sidebar.slider(
    "Hệ số e ngại rủi ro (γ):",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="γ càng cao, mô hình càng ưu tiên danh mục có rủi ro thấp (ít biến động), γ thấp ưu tiên danh mục lợi nhuận cao."
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Hệ thống sử dụng**:\n- **yfinance** để tải dữ liệu lịch sử.\n- **CVXPY** cho tối ưu hóa cổ điển.\n- **Qiskit Statevector** để mô phỏng mạch QAOA.")

# ----------------- Nội dung chính (Main) -----------------
# Header ấn tượng
st.markdown("""
<div class="main-header">
    <div class="quantum-badge">Quantum Tech & Financial Optimization Demo</div>
    <h1>QUANT PORTFOLIO</h1>
    <p>Tối ưu hóa danh mục đầu tư S&P500 bằng thuật toán Cổ điển (MVO) & Lượng tử (QAOA)</p>
</div>
""", unsafe_allow_html=True)

# Kiểm tra dữ liệu đầu vào
if len(selected_tickers) < 2:
    st.error("❌ Vui lòng chọn ít nhất 2 cổ phiếu để thực hiện tối ưu hóa danh mục đầu tư!")
    st.stop()

# Hàm tải dữ liệu được cache lại
@st.cache_data(show_spinner=False)
def load_data_cached(tickers, start, end):
    df = get_stock_data(tickers, start, end)
    if df.empty:
        # Nếu rỗng (lỗi tải cũ), tự động xóa bộ nhớ cache để buộc tải lại mạng ở lượt tiếp theo
        st.cache_data.clear()
    return df

with st.spinner("🚀 Đang tải dữ liệu giá cổ phiếu từ Yahoo Finance..."):
    # Chuyển đổi đối tượng ngày thành chuỗi dạng YYYY-MM-DD để tránh lỗi phân tích ngày của yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    prices_df = load_data_cached(selected_tickers, start_str, end_str)

if prices_df.empty:
    st.error("❌ Không thể tải dữ liệu cho các cổ phiếu đã chọn. Vui lòng kiểm tra lại mã cổ phiếu hoặc khoảng thời gian!")
    
    # Giao diện chẩn đoán lỗi chuyên sâu trực tiếp trên màn hình
    st.markdown("### 🔍 Trình chẩn đoán lỗi hệ thống (Diagnostics):")
    try:
        import yfinance as yf
        st.write(f"- Phiên bản yfinance trên Cloud: `{yf.__version__}`")
        st.write("- Đang thử tải kết nối kiểm tra (AAPL trong 5 ngày)...")
        test_df = yf.download("AAPL", period="5d")
        if test_df.empty:
            st.warning("⚠️ Máy chủ Yahoo Finance từ chối kết nối hoặc trả về bảng rỗng cho dải IP này!")
        else:
            st.success("✅ Tải kiểm tra AAPL thành công! Lỗi có thể do bộ nhớ cache chưa được làm sạch hoàn toàn hoặc lỗi định dạng khác.")
    except Exception as ex:
        st.exception(ex)
        
    st.stop()

# Tính toán tỷ suất lợi nhuận, mean returns và cov_matrix
daily_returns, mean_returns, cov_matrix = calculate_returns(prices_df)

# Phân chia Không gian Hiển thị bằng st.tabs
tab_explore, tab_classical, tab_quantum = st.tabs([
    "📊 Khám Phá Dữ Liệu", 
    "📈 Kết Quả Cổ Điển (MVO)", 
    "⚛️ Kết Quả Lượng Tử (QAOA)"
])

# ================= Giao diện Tab 1: Khám Phá Dữ Liệu =================
with tab_explore:
    st.subheader("🔍 Phân Tích Thống Kê & Biến Động Lịch Sử")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Thống kê mô tả (Mức độ lợi nhuận & rủi ro năm)")
        
        # Tạo bảng thống kê đẹp mắt
        stats_df = pd.DataFrame({
            'Giá cuối cùng ($)': prices_df.iloc[-1].round(2),
            'Lợi nhuận TB năm (%)': (mean_returns * 100).round(2),
            'Độ lệch chuẩn năm (%)': (daily_returns.std() * np.sqrt(252) * 100).round(2),
            'Hệ số Sharpe đơn lẻ': ((mean_returns - 0.02) / (daily_returns.std() * np.sqrt(252))).round(2)
        })
        
        st.dataframe(stats_df, use_container_width=True)
        
        st.markdown("""
        *Lưu ý: Hệ số Sharpe đơn lẻ được tính toán dựa trên lãi suất phi rủi ro mặc định là 2% ($R_f = 0.02$). Các số liệu được quy đổi theo năm (annualized).*
        """)
        
    with col2:
        st.markdown("### 🔗 Ma trận tương quan (Correlation Matrix)")
        
        corr_matrix = daily_returns.corr()
        
        # Vẽ Heatmap ma trận tương quan bằng Plotly
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            labels=dict(color="Correlation Coefficient")
        )
        fig_corr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#e2e8f0"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=320
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 Biến động tăng trưởng lũy kế của 100$ đầu tư ban đầu")
    
    # Tính toán Cumulative Return: (1 + r).cumprod() * 100
    cum_returns = (1 + daily_returns).cumprod() * 100
    # Thêm giá trị ban đầu là 100 ở dòng đầu
    first_row = pd.DataFrame(100.0, index=[prices_df.index[0]], columns=cum_returns.columns)
    cum_returns = pd.concat([first_row, cum_returns])
    
    # Biểu đồ Line Chart lũy kế Plotly
    fig_line = px.line(
        cum_returns,
        labels={"index": "Thời gian", "value": "Giá trị danh mục ($)"},
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    
    fig_line.update_layout(
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ================= Giao diện Tab 2: Kết Quả Cổ Điển =================
with tab_classical:
    st.subheader("📈 Tối ưu hóa Phương sai - Kỳ vọng (Mean-Variance Optimization)")
    
    # Chạy tính toán Efficient Frontier
    with st.spinner("⏳ Đang giải bài toán tối ưu hóa lồi CVXPY..."):
        random_portfolios, frontier_df, opt_portfolio = generate_efficient_frontier(
            mean_returns, cov_matrix, risk_aversion, num_portfolios=1000
        )
        
    # Trực quan hóa Thẻ KPI Danh mục tối ưu
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Lợi nhuận Kỳ vọng Năm</div>
            <div class="card-value">{opt_portfolio['return']*100:.2f}%</div>
            <div class="card-desc">Tỷ suất sinh lời kỳ sinh năm kỳ vọng</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi2:
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Rủi ro Danh mục (Volatility)</div>
            <div class="card-value">{opt_portfolio['volatility']*100:.2f}%</div>
            <div class="card-desc">Độ lệch chuẩn lợi nhuận năm</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_kpi3:
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">Hệ số Sharpe (Sharpe Ratio)</div>
            <div class="card-value">{opt_portfolio['sharpe']:.2f}</div>
            <div class="card-desc">Lợi nhuận vượt trội trên mỗi đơn vị rủi ro</div>
        </div>
        """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([1.1, 0.9])
    
    with col_chart1:
        st.markdown("### 🌌 Đường biên hiệu quả (Efficient Frontier)")
        
        # Vẽ biểu đồ Efficient Frontier bằng Plotly
        fig_frontier = ob.Figure()
        
        # 1. Scatter cloud cho các danh mục ngẫu nhiên
        fig_frontier.add_trace(ob.Scatter(
            x=random_portfolios['volatility'],
            y=random_portfolios['return'],
            mode='markers',
            marker=dict(
                size=5,
                color=random_portfolios['sharpe'],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="Sharpe Ratio",
                    thickness=15
                )
            ),
            name="Danh mục ngẫu nhiên",
            text=[f"Sharpe: {sr:.2f}<br>Lợi nhuận: {ret*100:.1f}%<br>Rủi ro: {vol*100:.1f}%" 
                  for sr, ret, vol in zip(random_portfolios['sharpe'], random_portfolios['return'], random_portfolios['volatility'])],
            hoverinfo='text'
        ))
        
        # 2. Vẽ đường biên tối ưu
        fig_frontier.add_trace(ob.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='lines',
            line=dict(color='#8b5cf6', width=3.5, dash='solid'),
            name="Đường biên Hiệu quả (Optimal Line)"
        ))
        
        # 3. Đánh dấu danh mục tối ưu hiện tại
        fig_frontier.add_trace(ob.Scatter(
            x=[opt_portfolio['volatility']],
            y=[opt_portfolio['return']],
            mode='markers',
            marker=dict(
                color='#fbbf24',
                size=18,
                symbol='star',
                line=dict(color='black', width=1.5)
            ),
            name="Danh mục Tối ưu hiện tại",
            text=[f"DỰ ÁN TỐI ƯU CỔ ĐIỂN<br>Sharpe: {opt_portfolio['sharpe']:.2f}<br>Lợi nhuận: {opt_portfolio['return']*100:.1f}%<br>Rủi ro: {opt_portfolio['volatility']*100:.1f}%"],
            hoverinfo='text'
        ))
        
        fig_frontier.update_layout(
            xaxis_title="Độ lệch chuẩn (Rủi ro năm)",
            yaxis_title="Lợi nhuận Kỳ vọng Năm",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#e2e8f0"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".1%"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickformat=".1%"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        
        st.plotly_chart(fig_frontier, use_container_width=True)
        
    with col_chart2:
        st.markdown("### 🍰 Tỷ trọng phân bổ vốn (Asset Allocation)")
        
        # Tạo dữ liệu tỷ trọng
        alloc_df = pd.DataFrame({
            'Cổ phiếu': selected_tickers,
            'Tỷ trọng (%)': (opt_portfolio['weights'] * 100)
        })
        
        # Loại bỏ các cổ phiếu có tỷ trọng quá nhỏ (<0.1%)
        alloc_df = alloc_df[alloc_df['Tỷ trọng (%)'] > 0.1].sort_values('Tỷ trọng (%)', ascending=False)
        
        # Biểu đồ hình quạt (Pie Chart) Plotly
        fig_pie = px.pie(
            alloc_df,
            values='Tỷ trọng (%)',
            names='Cổ phiếu',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#e2e8f0"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

# ================= Giao diện Tab 3: Kết Quả Lượng Tử =================
with tab_quantum:
    st.subheader("⚛️ Mô phỏng Tối ưu hóa Lượng tử QAOA (Quantum Approximate Optimization Algorithm)")
    
    st.markdown("""
    Thuật toán Lượng tử **QAOA** giải quyết bài toán tối ưu hóa danh mục đầu tư dưới dạng **lựa chọn nhị phân** (chọn hoặc không chọn một cổ phiếu) 
    sao cho thỏa mãn ràng buộc ngân sách (Budget - chọn đúng $B$ cổ phiếu tốt nhất) nhằm tối đa hóa lợi nhuận và giảm thiểu rủi ro hiệp phương sai.
    """)
    
    # Cấu hình tham số cho Lượng tử
    col_q1, col_q2, col_q3 = st.columns([1, 1, 1])
    
    with col_q1:
        q_budget = st.slider(
            "Ngân sách cổ phiếu muốn chọn (Budget B):",
            min_value=1,
            max_value=len(selected_tickers),
            value=min(3, len(selected_tickers)),
            step=1,
            help="Số lượng cổ phiếu chính xác bạn muốn chọn vào danh mục đầu tư lượng tử nhị phân."
        )
        
    with col_q2:
        q_layers = st.slider(
            "Số lớp của mạch QAOA (p):",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
            help="Số lớp tham số (layer depth) của mạch lượng tử. Lớp càng sâu, độ chính xác mô phỏng càng cao nhưng tính toán lâu hơn."
        )
        
    with col_q3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_quantum = st.button("⚛️ KHỞI CHẠY THUẬT TOÁN LƯỢNG TỬ", use_container_width=True, type="primary")

    if run_quantum or 'qaoa_results' in st.session_state:
        # Nếu nhấn nút, tính toán lại và lưu vào session state để tránh mất dữ liệu khi người dùng chuyển qua lại
        if run_quantum or 'qaoa_results' not in st.session_state:
            with st.spinner("⚛️ Đang khởi tạo Hamiltonian Ising, thiết lập mạch lượng tử và thực hiện tối ưu hóa cổ điển góc quay bằng thuật toán COBYLA..."):
                try:
                    qaoa_results, probs = solve_qaoa(
                        mean_returns, cov_matrix, risk_aversion, q_budget, p=q_layers
                    )
                    st.session_state['qaoa_results'] = qaoa_results
                    st.session_state['qaoa_probs'] = probs
                except Exception as e:
                    st.error(f"❌ Có lỗi trong quá trình mô phỏng lượng tử: {e}")
                    st.stop()
                    
        # Load dữ liệu từ session state
        qaoa_results = st.session_state['qaoa_results']
        probs = st.session_state['qaoa_probs']
        
        # Calculate penalty exactly as in utils.py to display in mathematical explanation
        cov_val = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix
        max_cov = np.max(cov_val)
        penalty = max(1.0, 2.0 * risk_aversion * max_cov)
        
        # Lấy trạng thái tốt nhất
        best_state = qaoa_results[0]
        selected_indices = np.where(best_state['binary_array'] == 1)[0]
        selected_stocks = [selected_tickers[i] for i in selected_indices]
        
        # 1. Thẻ KPI hiển thị danh mục lượng tử khuyến khích
        st.markdown("### 🎯 Kết quả Danh mục Lượng tử Khuyến nghị")
        

            
        # Chia cột hiển thị biểu đồ phân phối xác suất & so sánh danh mục
        col_qch1, col_qch2 = st.columns([1.1, 0.9])
        
        with col_qch1:
            st.markdown("### 📊 Phân phối xác suất đo được của các trạng thái Qubit (Top 8)")
            
            # Chuẩn bị dữ liệu vẽ biểu đồ phân phối
            top_states_df = pd.DataFrame(qaoa_results[:8])
            top_states_df['prob_pct'] = top_states_df['prob'] * 100
            
            # Biểu đồ hình cột Plotly
            fig_probs = px.bar(
                top_states_df,
                x='state_str',
                y='prob_pct',
                labels={'state_str': 'Trạng thái Qubit (Stock 0 -> Stock N-1)', 'prob_pct': 'Xác suất xuất hiện (%)'},
                color='prob_pct',
                color_continuous_scale='plasma'
            )
            
            fig_probs.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#e2e8f0"),
                xaxis=dict(type='category'),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=380,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_probs, use_container_width=True)
            st.markdown(f"*Chú ý: Chuỗi nhị phân (ví dụ `{best_state['state_str']}`) biểu diễn các cổ phiếu có được chọn (1) hoặc loại bỏ (0) theo thứ tự: {', '.join(selected_tickers)}.*")
            
        with col_qch2:
            st.markdown("### ⚖️ So sánh Tỷ trọng Cổ điển MVO vs Lượng tử QAOA")
            
            # Tạo DataFrame so sánh
            # Danh mục lượng tử phân bổ đều giữa các cổ phiếu được chọn
            q_weights = np.zeros(len(selected_tickers))
            if len(selected_indices) > 0:
                q_weights[selected_indices] = 1.0 / len(selected_indices)
                
            # Trọng số cổ điển
            c_weights = optimize_classical_mvo(mean_returns, cov_matrix, risk_aversion)
            
            compare_df = pd.DataFrame({
                'Cổ phiếu': selected_tickers,
                'Cổ điển MVO (%)': (c_weights * 100).round(2),
                'Lượng tử QAOA (%)': (q_weights * 100).round(2)
            })
            
            # Vẽ biểu đồ Grouped Bar Chart bằng Plotly
            fig_compare = ob.Figure()
            fig_compare.add_trace(ob.Bar(
                x=compare_df['Cổ phiếu'],
                y=compare_df['Cổ điển MVO (%)'],
                name='Cổ điển MVO (%)',
                marker_color='#4f46e5'
            ))
            fig_compare.add_trace(ob.Bar(
                x=compare_df['Cổ phiếu'],
                y=compare_df['Lượng tử QAOA (%)'],
                name='Lượng tử QAOA (%)',
                marker_color='#db2777'
            ))
            
            fig_compare.update_layout(
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#e2e8f0"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Tỷ trọng phân bổ (%)"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_compare, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Nguyên lý hoạt động & Công thức Toán học")
        
        st.markdown("""
        Để giải quyết bài toán này trên máy tính lượng tử, ta phải thực hiện quy trình ánh xạ và mô phỏng sau:
        
        #### 1. Ánh xạ về mô hình QUBO (Quadratic Unconstrained Binary Optimization)
        Ta thiết lập hàm chi phí cần giảm thiểu có dạng nhị phân với các biến $x_i \in \{0, 1\}$:
        """)
        
        st.latex(r"""
        C(x) = \frac{\gamma}{2} \sum_{i,j} \Sigma_{i,j} x_i x_j - \sum_i \mu_i x_i + \lambda \left(\sum_i x_i - B\right)^2
        """)
        
        st.markdown(f"""
        Trong đó:
        - $\Sigma_{{i,j}}$ là ma trận hiệp phương sai.
        - $\mu_i$ là lợi nhuận kỳ vọng của cổ phiếu $i$.
        - $\gamma$ là hệ số e ngại rủi ro (hiện đang đặt là **{risk_aversion}**).
        - $B$ là số lượng cổ phiếu cần chọn (hiện đang đặt là **{q_budget}**).
        - $\lambda$ là trọng số phạt vi phạm ràng buộc ngân sách (hiện đang tính toán tự động bằng **{penalty:.2f}**).
        
        #### 2. Chuyển đổi sang Hamiltonian Ising
        Bằng cách thay thế các biến nhị phân bằng toán tử Pauli Z qua phép biến đổi $x_i = \frac{{I - Z_i}}{{2}}$, ta thu được toán tử năng lượng Hamiltonian của hệ lượng tử:
        """)
        
        st.latex(r"""
        H_C = \sum_{i} h_i Z_i + \sum_{i < j} J_{i,j} Z_i Z_j
        """)
        
        st.markdown("""
        #### 3. Cấu trúc mạch lượng tử QAOA
        Mạch lượng tử bao gồm việc áp dụng liên tiếp các lớp tiến hóa (với chiều sâu $p$):
        1. **Khởi tạo**: Tạo trạng thái chồng chập đều bằng cách đặt tất cả qubit qua cổng Hadamard: $|+\rangle^{\otimes N}$.
        2. **Toán tử chi phí (Cost Unitary)**: Áp dụng $U(H_C, \gamma_k) = e^{-i \gamma_k H_C}$. Trong mạch, các số hạng $h_i Z_i$ tương ứng cổng xoay pha $R_z(2 \gamma_k h_i)$, còn $J_{i,j} Z_i Z_j$ tương ứng với cụm cổng CNOT và cổng $R_z(2 \gamma_k J_{i,j})$.
        3. **Toán tử trộn (Mixer Unitary)**: Áp dụng $U(H_B, \beta_k) = e^{-i \beta_k H_B}$ với $H_B = \sum_i X_i$, tương ứng với cổng xoay $R_x(2 \beta_k)$ trên mỗi qubit.
        """)
    else:
        # Hướng dẫn khi chưa chạy
        st.info("💡 Vui lòng thiết lập các tham số lượng tử ở trên và nhấn nút **⚛️ KHỞI CHẠY THUẬT TOÁN LƯỢNG TỬ** để xem kết quả mô phỏng mạch lượng tử QAOA.")
