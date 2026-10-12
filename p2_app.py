                        'ชนะเต็ม': GREEN, 'ชนะครึ่ง': GREEN_SOFT,
                        'เจ๊า': '#4b5f54', 'แพ้ครึ่ง': '#b8575b', 'แพ้เต็ม': RED
                    }
                    counts = df_comp['ผลเปรียบเทียบ'].value_counts()
                    order = [k for k in color_map if k in counts.index]
                    pie_fig = go.Figure(go.Pie(
                        labels=order, values=[counts[k] for k in order],
                        hole=0.72, sort=False, direction='clockwise',
                        marker=dict(colors=[color_map[k] for k in order], line=dict(color=BG, width=3)),
                        textinfo='none', hovertemplate='%{label}: %{value} คู่ (%{percent})<extra></extra>'
                    ))
                    base_layout(pie_fig, 380)
                    pie_fig.update_layout(
                        showlegend=True,
                        legend=dict(orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5,
                                    font=dict(color=TEXT, size=13)),
                        annotations=[
                            dict(text=f"<b>{win_rate}%</b>", x=0.5, y=0.55, showarrow=False,
                                 font=dict(size=34, color=TEXT)),
                            dict(text="Win Rate", x=0.5, y=0.43, showarrow=False,
                                 font=dict(size=13, color=MUTED)),
                        ]
                    )
                    st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})

                # --- 3. Bar แยกตามกฎ ---
                with col_chart2:
                    panel_header("อัตราชนะแยกตามกฎ", "เขียว ≥ 60%  ·  เหลือง 55-60%  ·  แดง < 55%")

                    if not df_stats.empty and 'อัตราชนะ' in df_stats.columns:
                        plot_df = df_stats.copy()
                        rule_col = plot_df.columns[0]
                        plot_df['WinRate_Val'] = pd.to_numeric(
                            plot_df['อัตราชนะ'].astype(str).str.replace('%', '').str.strip(),
                            errors='coerce').fillna(0)
                        plot_df = plot_df[plot_df['WinRate_Val'] > 0].sort_values('WinRate_Val', ascending=True)

                        bar_colors = [GREEN if v >= 60 else AMBER if v >= 55 else RED for v in plot_df['WinRate_Val']]

                        fig_bar = go.Figure(go.Bar(
                            x=plot_df['WinRate_Val'], y=plot_df[rule_col], orientation='h',
                            marker=dict(color=bar_colors, cornerradius=6),
                            text=[f"{v:g}%" for v in plot_df['WinRate_Val']], textposition='outside',
                            textfont=dict(color=TEXT, size=13), cliponaxis=False,
                            hovertemplate='%{y}: %{x}%<extra></extra>'
                        ))
                        base_layout(fig_bar, max(380, 46 * len(plot_df) + 40))
                        fig_bar.update_layout(
                            xaxis=dict(range=[0, 108], visible=False),
                            yaxis=dict(showgrid=False, title="", tickfont=dict(size=13, color=TEXT)),
                            bargap=0.45, showlegend=False,
                            margin=dict(l=10, r=40, t=10, b=10),
                        )
                        fig_bar.add_vline(x=55, line=dict(color=LINE, width=1.5, dash='dot'))
                        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("ยังไม่มีข้อมูลสถิติเพียงพอสำหรับสร้างกราฟ")

            # --- 4. การ์ดสถิติรายกฎ ---
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            panel_header("สถิติรายกฎ", "เรียงจากอัตราชนะสูงสุด · ลูกศรเทียบกับค่าก่อนหน้าที่ระบบบันทึกไว้")

            if not df_stats.empty and 'อัตราชนะ' in df_stats.columns:
                tbl = df_stats.copy()
                tbl['_v'] = pd.to_numeric(
                    tbl['อัตราชนะ'].astype(str).str.replace('%', '').str.strip(),
                    errors='coerce').fillna(0)
                tbl = tbl.sort_values('_v', ascending=False)
                rule_col = df_stats.columns[0]
                extra_cols = [c for c in df_stats.columns if c not in (rule_col, 'อัตราชนะ')]

                rates = {str(r[rule_col]).strip(): round(float(r['_v']), 2)
                         for _, r in tbl.iterrows() if r['_v'] > 0}
                history, hist_err = cached_rate_history(workbook, rates)
                if hist_err:
                    st.warning(f"บันทึกประวัติสถิติไม่สำเร็จ จึงยังแสดงลูกศรไม่ได้: {hist_err}")

                deltas = [trend_delta(history.get(n, [])) for n in rates]
                n_up = sum(1 for d in deltas if d is not None and d > 0)
                n_down = sum(1 for d in deltas if d is not None and d < 0)
                if n_up or n_down:
                    chips = []
                    if n_up: chips.append(f"<span class='trend up'>▲ เพิ่มขึ้น {n_up} กฎ</span>")
                    if n_down: chips.append(f"<span class='trend down'>▼ ลดลง {n_down} กฎ</span>")
                    st.markdown(f"<div class='trend-sum'>{''.join(chips)}</div>", unsafe_allow_html=True)

                cards = ["<div class='rule-grid'>"]
                for _, row in tbl.iterrows():
                    v = row['_v']
                    name = str(row[rule_col]).strip()
                    k, tag = ("g", "แนะนำ") if v >= 60 else ("a", "ระวัง") if v >= 55 else ("r", "หลีกเลี่ยง")
                    meta = "".join(f"<span>{esc(str(c))}<b>{esc(str(row[c]))}</b></span>" for c in extra_cols
                                   if str(row[c]).strip() != "")
                    cards.append(
                        f"<div class='rule-card'>"
                        f"<div class='rule-top'><span class='rule-name'>{esc(name)}</span>"
                        f"<span class='rule-tag tag-{k}'>{tag}</span></div>"
                        f"<div class='rule-rate-row'><div class='rule-rate t-{k}'>{esc(str(row['อัตราชนะ']))}</div>"
                        f"{trend_html(history.get(name, []))}</div>"
                        f"<div class='rule-bar'><div class='b-{k}' style='width:{min(v, 100)}%'></div></div>"
                        f"<div class='rule-meta'>{meta}</div>"
                        f"</div>")
                cards.append("</div>")
                st.markdown("".join(cards), unsafe_allow_html=True)
            else:
                st.warning("ไม่พบข้อมูลสถิติในตาราง")

        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูลสถิติ: {str(e)}")
