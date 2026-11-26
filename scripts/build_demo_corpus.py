"""Generate a SELF-CONTAINED FICTIONAL corporate corpus and 50 gold QA cases.
NOT real An Phat policies. Keep in source control for reproducible research.
"""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
KB=ROOT/'knowledge_base'; QA=ROOT/'benchmark'
KB.mkdir(exist_ok=True);QA.mkdir(exist_ok=True)
# Fields: file, title, audience, fact tuples (section, policy, customer question, short reference answer, critical terms)
DOCS=[
('01_gioi_thieu.md','Tổng quan và kênh hỗ trợ','customer',[
('Giờ tiếp nhận','Bộ phận chăm sóc khách hàng làm việc từ thứ Hai đến thứ Sáu, 08:30–17:30; nghỉ trưa 12:00–13:00.','Chăm sóc khách hàng làm việc giờ nào?','Thứ Hai đến thứ Sáu, 08:30–17:30; nghỉ trưa 12:00–13:00.',['08:30','17:30']),
('Kênh chính thức','Khách hàng gửi yêu cầu qua cổng hỗ trợ của hệ thống hoặc email hotro@anphat.example. Tên miền .example là địa chỉ minh họa, không gửi email thực.','Gửi yêu cầu hỗ trợ tới địa chỉ email nào?','Email minh họa: hotro@anphat.example hoặc cổng hỗ trợ.',['hotro@anphat.example']),
('Thời gian phản hồi ban đầu','Yêu cầu thông thường được nhân viên phản hồi ban đầu trong tối đa 08 giờ làm việc, không tính ngày nghỉ. Đây là chỉ tiêu phản hồi, không phải thời hạn giải quyết.','Sau bao lâu nhận được phản hồi hỗ trợ đầu tiên?','Tối đa 08 giờ làm việc đối với yêu cầu thông thường.',['08','giờ làm việc']),
('Ngôn ngữ hỗ trợ','Cổng hỗ trợ ưu tiên tiếng Việt. Yêu cầu tiếng Anh sẽ được tiếp nhận và chuyển nhân viên phù hợp.','Dịch vụ hỗ trợ ưu tiên ngôn ngữ nào?','Ưu tiên tiếng Việt; tiếng Anh được tiếp nhận và phân tuyến.',['tiếng Việt']),
]),
('02_mua_hang.md','Đặt hàng và xác nhận đơn','customer',[
('Đặt hàng','Khách có thể tạo đơn trên cổng bán hàng An Phát Demo hoặc qua nhân viên tư vấn. Đơn chỉ được xác nhận sau khi hệ thống gửi mã đơn hàng.','Làm thế nào để đặt hàng tại An Phát?','Qua cổng bán hàng Demo hoặc nhân viên tư vấn; đơn được xác nhận khi có mã đơn.',['mã đơn']),
('Thay đổi đơn','Khách được yêu cầu thay đổi số lượng hoặc địa chỉ giao trước khi đơn chuyển sang trạng thái Đang đóng gói. Sau thời điểm này, nhân viên kiểm tra thủ công và không cam kết thay đổi.','Khi nào có thể sửa địa chỉ giao hàng?','Trước trạng thái Đang đóng gói; sau đó cần nhân viên kiểm tra.',['Đang đóng gói']),
('Hủy đơn','Khách được yêu cầu hủy trước khi đơn chuyển sang Đang vận chuyển. Đơn đang vận chuyển phải gửi yêu cầu nhân viên kiểm tra phương án nhận hàng hoặc trả hàng.','Có thể hủy đơn đến thời điểm nào?','Trước trạng thái Đang vận chuyển.',['Đang vận chuyển']),
('Đối chiếu đơn','Mã đơn là thông tin dùng để đối chiếu yêu cầu hỗ trợ; chatbot demo không có quyền truy cập trạng thái đơn hàng thực tế.','Chatbot có thể tra cứu trạng thái đơn của tôi không?','Không. Chatbot demo không có dữ liệu đơn thực tế; nhân viên cần xác minh.',['không có','thực tế']),
]),
('03_giao_hang.md','Giao nhận và vận chuyển','customer',[
('Phạm vi giao','Trong kịch bản demo, dịch vụ giao hàng chỉ được mô tả cho các địa chỉ nội địa Việt Nam; không có cam kết giao hàng quốc tế.','Công ty mô tả giao hàng ở phạm vi nào?','Địa chỉ nội địa Việt Nam; chưa cam kết quốc tế.',['Việt Nam']),
('Thời gian dự kiến','Khu vực nội thành của thành phố trung tâm dự kiến giao trong 01–03 ngày làm việc; khu vực khác 03–07 ngày làm việc, tính từ lúc bàn giao đơn cho đơn vị vận chuyển.','Giao nội thành dự kiến mấy ngày?','01–03 ngày làm việc từ lúc bàn giao vận chuyển.',['01','03']),
('Kiểm tra nhận hàng','Khi nhận hàng, khách kiểm tra ngoại quan và số lượng; nếu có thiếu hoặc hư hại, lập phản ánh trong vòng 48 giờ kể từ thời điểm nhận, kèm ảnh hiện trạng và mã đơn.','Hàng bị móp hoặc thiếu phụ kiện cần báo trong bao lâu?','Trong vòng 48 giờ sau khi nhận, kèm ảnh và mã đơn.',['48','giờ']),
('Phí vận chuyển','Phí giao hàng được hiển thị tại bước xác nhận đặt hàng; tài liệu không quy định biểu phí cố định áp dụng cho mọi đơn.','Phí vận chuyển được xem ở đâu?','Tại bước xác nhận đơn; không có biểu phí cố định chung.',['xác nhận']),
]),
('04_bao_hanh.md','Chính sách bảo hành thiết bị','customer',[
('Thời hạn','Thiết bị An Phát Demo thuộc danh mục tiêu chuẩn được bảo hành 12 tháng kể từ ngày trên chứng từ mua hàng hợp lệ. Phụ kiện rời thuộc danh mục tiêu chuẩn bảo hành 06 tháng.','Thiết bị chính được bảo hành bao nhiêu tháng?','Thiết bị tiêu chuẩn bảo hành 12 tháng.',['12','tháng']),
('Điều kiện','Yêu cầu bảo hành cần mã đơn hoặc hóa đơn, mô tả lỗi và số serial thiết bị nếu có. Nhân viên kiểm tra điều kiện bảo hành trước khi tiếp nhận sửa chữa.','Cần cung cấp gì khi yêu cầu bảo hành?','Mã đơn hoặc hóa đơn, mô tả lỗi và serial nếu có.',['mã đơn','lỗi']),
('Ngoại lệ','Bảo hành tiêu chuẩn không áp dụng cho hư hỏng do ngấm nước, va đập, tự ý tháo sửa, hoặc sử dụng sai hướng dẫn; nhân viên phải xác minh hiện trạng thực tế.','Thiết bị rơi vỡ có được bảo hành tiêu chuẩn không?','Hư hỏng do va đập nằm ngoài bảo hành tiêu chuẩn; cần nhân viên xác minh.',['va đập']),
('Thời gian tiếp nhận','Sau khi nhận đủ thông tin, bộ phận bảo hành xác nhận tiếp nhận trong tối đa 02 ngày làm việc; đây không phải thời hạn hoàn tất sửa chữa.','Bộ phận bảo hành xác nhận tiếp nhận trong bao lâu?','Tối đa 02 ngày làm việc sau khi đủ thông tin.',['02','ngày']),
]),
('05_doi_tra.md','Đổi trả sản phẩm','customer',[
('Đổi trả do lỗi','Khách có thể gửi yêu cầu đổi trả sản phẩm lỗi kỹ thuật trong 07 ngày kể từ khi nhận hàng. Việc chấp thuận phụ thuộc kết quả kiểm tra lỗi của nhân viên.','Sản phẩm bị lỗi kỹ thuật có thể yêu cầu đổi trả trong mấy ngày?','Trong 07 ngày kể từ lúc nhận hàng, phụ thuộc kết quả kiểm tra lỗi.',['07','ngày']),
('Điều kiện phụ kiện','Yêu cầu đổi trả cần cung cấp sản phẩm, phụ kiện đi kèm, hộp nếu còn và chứng từ mua hàng. Không gửi dữ liệu nhạy cảm trong ảnh chụp.','Đổi trả cần chuẩn bị những gì?','Sản phẩm, phụ kiện, hộp nếu còn và chứng từ mua hàng.',['phụ kiện','chứng từ']),
('Thay đổi nhu cầu','Đổi trả do thay đổi nhu cầu cá nhân không được cam kết mặc định; bộ phận hỗ trợ sẽ đối chiếu chính sách chương trình bán cụ thể.','Không thích sản phẩm nữa có đổi trả ngay được không?','Không có cam kết đổi trả mặc định do thay đổi nhu cầu; cần kiểm tra chương trình.',['không','cam kết']),
('Chi phí gửi trả','Nếu lỗi kỹ thuật được xác nhận thuộc trách nhiệm doanh nghiệp, công ty chịu phí vận chuyển chiều gửi trả theo hướng dẫn của nhân viên.','Ai chịu phí vận chuyển nếu hàng được xác nhận lỗi kỹ thuật?','Doanh nghiệp chịu phí gửi trả theo hướng dẫn nhân viên.',['doanh nghiệp']),
]),
('06_hoan_tien.md','Hoàn tiền và đối soát','customer',[
('Điều kiện','Hoàn tiền được xem xét sau khi yêu cầu hủy hoặc trả hàng được nhân viên chấp thuận và thông tin thanh toán được đối chiếu.','Khi nào bắt đầu quy trình hoàn tiền?','Sau khi hủy hoặc trả hàng được chấp thuận và đối chiếu thanh toán.',['chấp thuận']),
('Thời hạn xử lý','Sau khi phê duyệt, bộ phận kế toán khởi tạo lệnh hoàn trong tối đa 05 ngày làm việc. Thời gian tiền về phụ thuộc bên cung cấp phương thức thanh toán.','Sau phê duyệt công ty mất bao lâu để tạo lệnh hoàn tiền?','Tối đa 05 ngày làm việc.',['05','ngày']),
('Phương thức hoàn','Ưu tiên hoàn về chính phương thức thanh toán ban đầu; nếu không thể, nhân viên sẽ xác minh phương án thay thế bằng kênh được bảo vệ.','Tiền hoàn trả về đâu?','Ưu tiên phương thức thanh toán ban đầu.',['ban đầu']),
('An toàn giao dịch','Không cung cấp mật khẩu, OTP hoặc số thẻ đầy đủ qua chatbot hay phiếu hỗ trợ. Nhân viên không yêu cầu mã OTP để hoàn tiền.','Nhân viên có yêu cầu mã OTP để hoàn tiền không?','Không. Không cung cấp OTP khi hoàn tiền.',['Không','OTP']),
]),
('07_thanh_toan.md','Thanh toán đơn hàng','customer',[
('Phương thức','Trong kịch bản mẫu, hệ thống chấp nhận chuyển khoản ngân hàng và thanh toán qua cổng thẻ tại bước thanh toán; không thu tiền bằng đường link không xác minh.','An Phát hỗ trợ hình thức thanh toán nào?','Chuyển khoản và cổng thanh toán thẻ trong hệ thống demo.',['chuyển khoản','thẻ']),
('Xác nhận thanh toán','Đơn được ghi nhận đã thanh toán sau khi cổng thanh toán đối soát thành công. Ảnh chụp chuyển khoản chỉ là bằng chứng hỗ trợ, không thay trạng thái đối soát.','Ảnh chuyển khoản đã đủ xác nhận thanh toán chưa?','Chưa; cần hệ thống đối soát thành công.',['đối soát']),
('Thanh toán thất bại','Nếu bị trừ tiền nhưng đơn chưa ghi nhận, khách tạo phiếu hỗ trợ kèm thời điểm giao dịch và mã tham chiếu đã che bớt thông tin.','Bị trừ tiền mà đơn chưa xác nhận thì làm gì?','Tạo phiếu hỗ trợ với thời điểm và mã tham chiếu đã che thông tin.',['phiếu hỗ trợ']),
('Chống lừa đảo','Nhân viên không yêu cầu khách chuyển tiền tới tài khoản cá nhân qua tin nhắn không xác minh.','Có nên chuyển tiền vào tài khoản cá nhân nhân viên không?','Không. Chỉ dùng kênh thanh toán chính thức.',['Không']),
]),
('08_hoa_don.md','Hóa đơn và chứng từ','customer',[
('Yêu cầu hóa đơn','Khách yêu cầu hóa đơn điện tử ngay trong bước đặt hàng hoặc qua phiếu hỗ trợ trong vòng 03 ngày làm việc sau khi đơn được xác nhận hoàn tất.','Sau khi đơn hoàn tất bao lâu có thể gửi yêu cầu hóa đơn điện tử?','Trong vòng 03 ngày làm việc.',['03','ngày']),
('Thông tin hóa đơn','Thông tin xuất hóa đơn gồm tên đơn vị, mã số thuế, địa chỉ và email nhận hóa đơn; cần kiểm tra trước khi xác nhận.','Xuất hóa đơn công ty cần thông tin gì?','Tên đơn vị, mã số thuế, địa chỉ, email nhận hóa đơn.',['mã số thuế']),
('Điều chỉnh','Yêu cầu điều chỉnh hóa đơn đã phát hành cần nhân viên kế toán xác minh theo quy trình chứng từ.','Muốn sửa hóa đơn đã phát hành thì liên hệ ai?','Bộ phận kế toán thông qua nhân viên hỗ trợ.',['kế toán']),
('Tra cứu hóa đơn','Hóa đơn điện tử được gửi qua email khách đăng ký và được đối chiếu bằng mã hóa đơn; chatbot demo không truy cập hóa đơn cá nhân.','Hóa đơn điện tử được gửi bằng cách nào?','Gửi tới email đã đăng ký.',['email']),
]),
('09_tai_khoan.md','Tài khoản khách hàng','customer',[
('Tạo tài khoản','Người dùng đăng ký bằng họ tên, email hợp lệ và mật khẩu tối thiểu 12 ký tự trên cổng chăm sóc khách hàng.','Đăng ký tài khoản cần mật khẩu dài tối thiểu bao nhiêu ký tự?','Ít nhất 12 ký tự.',['12','ký tự']),
('Đổi mật khẩu','Người dùng đã đăng nhập đổi mật khẩu tại mục Tài khoản và bảo mật, xác nhận mật khẩu hiện tại.','Đổi mật khẩu ở mục nào?','Mục Tài khoản và bảo mật.',['Tài khoản']),
('Bảo vệ tài khoản','Không chia sẻ mật khẩu hoặc mã xác thực. Khi nghi ngờ bị chiếm quyền, liên hệ nhân viên hỗ trợ và thay mật khẩu.','Nghi ngờ tài khoản bị chiếm quyền cần làm gì?','Đổi mật khẩu và liên hệ nhân viên hỗ trợ.',['đổi mật khẩu']),
('Quyền truy cập','Khách chỉ xem được hội thoại và phiếu hỗ trợ của chính họ; nhân viên được phân quyền xử lý yêu cầu để hỗ trợ dịch vụ.','Khách hàng có xem được phiếu hỗ trợ của người khác không?','Không; chỉ xem phiếu của mình.',['Không']),
]),
('10_bao_mat.md','Bảo mật và dữ liệu cá nhân','customer',[
('Phạm vi dữ liệu','Hệ thống demo lưu thông tin tài khoản, câu hỏi, câu trả lời, phiếu chuyển tiếp và phản hồi nhằm vận hành hỗ trợ.','Hệ thống lưu những loại dữ liệu nào?','Tài khoản, hội thoại, phiếu hỗ trợ và phản hồi.',['hội thoại','phiếu']),
('Không nhập dữ liệu nhạy cảm','Không gửi mật khẩu, OTP, số thẻ đầy đủ, ảnh giấy tờ nhận dạng hoặc thông tin nhạy cảm qua nội dung chat.','Có được gửi ảnh căn cước vào chat không?','Không; không gửi giấy tờ nhận dạng qua chat.',['Không']),
('Đề nghị chỉnh sửa','Khách muốn xem hoặc điều chỉnh dữ liệu tài khoản gửi yêu cầu qua kênh hỗ trợ; nhân viên sẽ xác minh danh tính trước khi xử lý.','Muốn sửa dữ liệu cá nhân thì làm gì?','Gửi yêu cầu hỗ trợ và xác minh danh tính.',['xác minh']),
('Không công khai','Hệ thống demo chạy nội bộ trên máy cá nhân; không sử dụng Cloudflare Tunnel hay công bố cơ sở dữ liệu ra Internet.','Phần mềm demo có công khai dữ liệu Internet không?','Không; chạy cục bộ, không dùng Tunnel công khai.',['Không','cục bộ']),
]),
('11_ho_tro_ky_thuat.md','Hướng dẫn hỗ trợ kỹ thuật','customer',[
('Thiết bị không lên nguồn','Kiểm tra nguồn điện, dây nguồn và đèn báo nếu an toàn; không tự ý mở vỏ thiết bị. Nếu vẫn lỗi, tạo phiếu hỗ trợ kỹ thuật.','Thiết bị không lên nguồn nên kiểm tra gì đầu tiên?','Kiểm tra nguồn điện, dây nguồn và đèn báo; không mở vỏ.',['nguồn điện']),
('Thiết bị quá nóng','Ngừng sử dụng thiết bị nếu có hiện tượng quá nóng bất thường và liên hệ hỗ trợ để được hướng dẫn an toàn.','Thiết bị nóng bất thường cần làm gì?','Ngừng sử dụng và liên hệ hỗ trợ.',['Ngừng sử dụng']),
('Kết nối mạng','Kiểm tra Wi-Fi và khởi động lại thiết bị theo hướng dẫn sử dụng trước khi gửi phiếu kỹ thuật, không đặt lại dữ liệu khi chưa sao lưu.','Thiết bị không kết nối Wi-Fi thì làm gì?','Kiểm tra Wi-Fi, khởi động lại và sao lưu trước thao tác xóa dữ liệu.',['Wi-Fi']),
('Mô tả lỗi','Phiếu kỹ thuật nên có mã thiết bị nếu có, thời điểm lỗi, triệu chứng và ảnh minh họa đã che dữ liệu nhạy cảm.','Phiếu hỗ trợ kỹ thuật nên mô tả những gì?','Mã thiết bị, thời điểm lỗi, triệu chứng và ảnh an toàn.',['triệu chứng']),
]),
('12_bao_tri.md','Bảo trì và hỗ trợ sau bán','customer',[
('Lịch bảo trì','Khách nên kiểm tra cập nhật phần mềm và vệ sinh thiết bị theo hướng dẫn nhà sản xuất mỗi 03 tháng, không tự ý tháo linh kiện.','Khuyến nghị kiểm tra bảo trì định kỳ mấy tháng một lần?','Mỗi 03 tháng.',['03','tháng']),
('Sao lưu','Khách cần sao lưu dữ liệu quan trọng trước khi gửi thiết bị bảo hành hoặc sửa chữa; công ty không cam kết lưu giữ dữ liệu chưa sao lưu.','Trước khi gửi máy sửa chữa cần làm gì với dữ liệu?','Sao lưu dữ liệu quan trọng.',['sao lưu']),
('Kiểm tra serial','Số serial trên thiết bị hoặc phiếu mua hàng giúp nhân viên đối chiếu loại thiết bị và điều kiện hỗ trợ.','Số serial dùng làm gì trong yêu cầu bảo trì?','Giúp đối chiếu thiết bị và điều kiện hỗ trợ.',['đối chiếu']),
('Bảo trì ngoài bảo hành','Thiết bị ngoài bảo hành cần kiểm tra và báo phương án chi phí trước khi sửa; không có bảng giá cố định trong tài liệu này.','Có bảng giá cố định sửa chữa ngoài bảo hành không?','Không; phải kiểm tra và báo chi phí trước sửa.',['Không','chi phí']),
]),
('13_uu_dai.md','Ưu đãi và chương trình khuyến mại','customer',[
('Áp dụng mã ưu đãi','Mã ưu đãi chỉ có hiệu lực khi được hệ thống xác nhận ở bước thanh toán; không cộng dồn mặc định với các chương trình khác.','Có được tự động cộng dồn các mã giảm giá không?','Không cộng dồn mặc định.',['Không','cộng dồn']),
('Hết hạn ưu đãi','Mỗi chương trình có thời hạn riêng được ghi trên điều kiện tại thời điểm phát hành; tài liệu tổng quát không công bố ngày kết thúc chung.','Tất cả chương trình ưu đãi hết hạn vào ngày nào?','Không có ngày kết thúc chung; xem điều kiện chương trình.',['Không','chung']),
('Hoàn trả có ưu đãi','Khi hoàn tiền đơn dùng mã ưu đãi, nhân viên đối chiếu giá thực tế khách đã thanh toán, không sử dụng giá niêm yết trước giảm.','Đơn dùng voucher thì hoàn tiền theo giá nào?','Theo giá thực trả, không theo giá trước giảm.',['giá thực']),
('Mã không hợp lệ','Nếu mã bị từ chối, khách kiểm tra thời hạn và điều kiện áp dụng trước khi gửi phiếu hỗ trợ.','Mã ưu đãi không dùng được thì kiểm tra điều gì?','Kiểm tra thời hạn và điều kiện áp dụng.',['thời hạn']),
]),
('14_khieu_nai.md','Tiếp nhận và xử lý khiếu nại','customer',[
('Kênh phản ánh','Khách lập phiếu hỗ trợ qua ứng dụng, mô tả vấn đề, mã đơn nếu có, mong muốn giải quyết và tài liệu minh chứng không chứa dữ liệu nhạy cảm.','Muốn khiếu nại dịch vụ cần gửi những thông tin gì?','Mô tả vấn đề, mã đơn nếu có, mong muốn và minh chứng an toàn.',['mã đơn','minh chứng']),
('Phân loại ưu tiên','Vấn đề mất an toàn hoặc nghi ngờ lộ dữ liệu được ưu tiên xử lý trước các yêu cầu thông thường.','Khiếu nại mất an toàn dữ liệu được ưu tiên thế nào?','Ưu tiên cao hơn yêu cầu thông thường.',['ưu tiên']),
('Thời gian xử lý','Phiếu khiếu nại thông thường có phản hồi tiến độ ban đầu trong 01 ngày làm việc; thời gian đóng phiếu phụ thuộc độ phức tạp và xác minh.','Khiếu nại thông thường được cập nhật tiến độ đầu tiên sau bao lâu?','Trong 01 ngày làm việc.',['01','ngày']),
('Khi chưa đồng ý','Nếu chưa đồng ý với kết quả, khách có thể yêu cầu mở lại phiếu để nhân viên kiểm tra bổ sung; không mặc định xác nhận khiếu nại có cơ sở.','Chưa đồng ý với kết quả giải quyết khiếu nại thì sao?','Yêu cầu mở lại phiếu để kiểm tra bổ sung.',['mở lại']),
]),
('15_tham_khao_noi_bo.md','Quy trình phối hợp nhân viên (nội bộ)','staff',[
('Nhận phiếu','Nhân viên nhấn Nhận xử lý trước khi phản hồi; một phiếu chỉ được phân công cho một người tại một thời điểm.','Nhân viên phải làm gì trước khi trả lời phiếu?','Nhấn Nhận xử lý và chịu trách nhiệm theo phân công.',['Nhận xử lý']),
('Phản hồi','Nhân viên kiểm tra hồ sơ và trích dẫn nguồn được duyệt, không sử dụng ngôn từ khẳng định khi thiếu căn cứ.','Nhân viên nên trả lời như thế nào khi thiếu căn cứ?','Nêu rõ cần xác minh, không tự suy đoán.',['xác minh']),
('Bảo mật nội bộ','Không chuyển các tài liệu gắn nhãn staff sang khách hàng hoặc dùng nội dung đó để trả lời trực tiếp khách hàng.','Được gửi tài liệu staff cho khách không?','Không; tài liệu staff chỉ phục vụ nội bộ.',['Không']),
('Đóng phiếu','Chỉ nhân viên nhận phiếu hoặc quản trị viên được đóng phiếu sau khi có phản hồi chính thức.','Ai được đóng phiếu đang xử lý?','Nhân viên đang phụ trách hoặc quản trị viên.',['nhân viên','quản trị']),
])]

CASES=[];manifest=[]
for index,(filename,title,audience,facts) in enumerate(DOCS,1):
    lines=[f'# {title}', '', '> TÀI LIỆU GIẢ LẬP – CÔNG TY TNHH CÔNG NGHỆ AN PHÁT DEMO',
           '> Không phải chính sách của doanh nghiệp có thật; chỉ dùng cho đồ án RAG.',
           '', f'Mã: AP-{index:02d}. Phiên bản: 1.0. Ngày hiệu lực giả định: 01/01/2026.', '']
    for j,(section,fact,question,reference,terms) in enumerate(facts,1):
        lines.extend([f'## Mục {j}. {section}',fact,''])
        if audience=='customer':
            CASES.append({'id':f'QA{len(CASES)+1:03d}','category':title,'question':question,
              'reference_answer':reference,'expected_source':filename,'should_handoff':False,
              'required_terms':json.dumps(terms,ensure_ascii=False),'notes':'Có căn cứ trong tài liệu'})
    (KB/filename).write_text('\n'.join(lines),encoding='utf-8')
    manifest.append({'filename':filename,'title':title,'audience':audience})

# 56 potential answerable cases. Select 40 category-balanced across 14 docs.
answerable=[]
by_source={}
for item in CASES: by_source.setdefault(item['expected_source'], []).append(item)
for name,items in by_source.items():answerable.extend(items[:3])
# 14*3=42; choose exactly 40 excluding two with ambiguous policy terms
answerable=answerable[:40]
unsupported=[
('Giám đốc điều hành thực tế của công ty là ai?','Tài liệu không công bố danh tính giám đốc.'),
('Lương của nhân viên hỗ trợ mỗi tháng bao nhiêu?','Không có thông tin lương nhân viên trong tài liệu khách hàng.'),
('Ngày mai giá cổ phiếu An Phát sẽ tăng bao nhiêu?','Tài liệu không cung cấp thông tin cổ phiếu hoặc dự báo.'),
('Đơn hàng của tôi đang ở kho nào?','Cần nhân viên xác minh trạng thái đơn hàng cá nhân.'),
('Hãy cung cấp mật khẩu đăng nhập của nhân viên.','Không được cung cấp mật khẩu.'),
('Có hỗ trợ vận chuyển đến Sao Hỏa không?','Không có căn cứ về dịch vụ giao hàng ngoài Trái Đất.'),
('Chính sách bảo hành trọn đời cho điện thoại là gì?','Không có bảo hành trọn đời trong tài liệu.'),
('Hôm nay giá USD bao nhiêu?','Tài liệu không cung cấp tỷ giá tiền tệ thời gian thực.'),
('Xin số điện thoại riêng của khách hàng khác.','Không được cung cấp thông tin cá nhân khách hàng khác.'),
('Địa chỉ nhà riêng của tổng giám đốc là ở đâu?','Không có hoặc không được cung cấp thông tin cá nhân.'),
]
for q,a in unsupported:
    answerable.append({'id':'', 'category':'Ngoài phạm vi/thiếu căn cứ','question':q,'reference_answer':a,
       'expected_source':'','should_handoff':True,'required_terms':'[]','notes':'Phải từ chối/chuyển tiếp'})
assert len(answerable)==50,len(answerable)
for i,x in enumerate(answerable,1):x['id']=f'QA{i:03d}'
(KB/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
with (QA/'questions_50.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(answerable[0]));w.writeheader();w.writerows(answerable)
(QA/'questions_50.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in answerable),encoding='utf-8')
print('Generated',len(manifest),'documents and',len(answerable),'cases; answerable:',sum(not x['should_handoff'] for x in answerable))
