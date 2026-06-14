/**
 * フォーム自動保存スクリプト。
 *
 * `data-autosave` 属性を持つ <form> を対象に、フィールドの変更を検知して
 * フォーム全体を action へ POST する（Save ボタンを押さずに即時反映する）。
 *
 * - テキスト入力は入力停止後 debounce、選択・チェック・ファイルは即時保存する。
 * - 既存のフォーム送信ハンドラをそのまま再利用する（X-Requested-With ヘッダで
 *   AJAX 判定し、サーバは JSON を返す）。
 * - `data-no-autosave` を持つ要素の変更は無視する（クロップ前の生画像など）。
 * - 画像クロップ確定など、イベントを伴わない変更は window.chotgorAutosave() で
 *   明示的に保存をトリガーできる。
 */
(function () {
  "use strict";

  /** テキスト入力の保存を遅延させる時間（ミリ秒）。 */
  var DEBOUNCE_MS = 700;

  /** 画面右下に出す保存ステータス表示要素（遅延生成）。 */
  var statusEl = null;

  /** 保存ステータス表示要素を取得する（無ければ生成する）。 */
  function ensureStatus() {
    if (statusEl) return statusEl;
    statusEl = document.createElement("div");
    statusEl.className = "autosave-status";
    document.body.appendChild(statusEl);
    return statusEl;
  }

  /**
   * 保存ステータスを表示する。
   * @param {string} text 表示文言
   * @param {string} kind saving | saved | error | invalid
   */
  function showStatus(text, kind) {
    var el = ensureStatus();
    el.textContent = text;
    el.dataset.kind = kind;
    el.classList.add("visible");
    clearTimeout(el._hideTimer);
    // 完了系の表示は一定時間で自動的に消す。
    if (kind === "saved" || kind === "invalid" || kind === "error") {
      el._hideTimer = setTimeout(function () {
        el.classList.remove("visible");
      }, 2200);
    }
  }

  /**
   * 1 つのフォームに自動保存の挙動を取り付ける。
   * @param {HTMLFormElement} form
   */
  function attach(form) {
    var timer = null;
    var saving = false;
    var pending = false;

    /** フォーム内容を POST する。保存中の多重実行は pending に畳む。 */
    function save() {
      if (saving) {
        pending = true;
        return;
      }
      // 不正な入力なら保存しない（ブラウザの検証 + setCustomValidity を流用）。
      // 最初の不正フィールドの validationMessage を出す（無ければ既定文言）。
      if (typeof form.checkValidity === "function" && !form.checkValidity()) {
        var invalid = form.querySelector(":invalid");
        var msg = (invalid && invalid.validationMessage) || "必須項目が未入力です";
        showStatus(msg, "invalid");
        return;
      }
      saving = true;
      showStatus("保存中…", "saving");
      fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { "X-Requested-With": "fetch" },
      })
        .then(function (res) {
          if (!res.ok) throw new Error("HTTP " + res.status);
          showStatus("保存しました", "saved");
        })
        .catch(function () {
          showStatus("保存に失敗しました", "error");
        })
        .then(function () {
          saving = false;
          if (pending) {
            pending = false;
            save();
          }
        });
    }

    /**
     * 保存をスケジュールする。
     * @param {boolean} immediate true なら debounce せず即時保存する
     */
    function schedule(immediate) {
      clearTimeout(timer);
      if (immediate) {
        save();
      } else {
        timer = setTimeout(save, DEBOUNCE_MS);
      }
    }

    // テキスト入力中は debounce 保存する。
    form.addEventListener("input", function (e) {
      if (e.target.closest("[data-no-autosave]")) return;
      schedule(false);
    });

    // 選択・チェック・ファイルは確定操作なので即時保存する。
    form.addEventListener("change", function (e) {
      if (e.target.closest("[data-no-autosave]")) return;
      var t = e.target;
      var immediate =
        t.tagName === "SELECT" ||
        t.type === "checkbox" ||
        t.type === "radio" ||
        t.type === "file";
      schedule(immediate);
    });

    // 画像クロップ確定など、イベントを伴わない変更からの明示トリガー。
    form.addEventListener("autosave:now", function () {
      schedule(true);
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var forms = document.querySelectorAll("form[data-autosave]");
    for (var i = 0; i < forms.length; i++) attach(forms[i]);
  });

  /**
   * 外部スクリプトから保存を明示的にトリガーする。
   * @param {HTMLFormElement|string} formOrSelector フォーム要素か CSS セレクタ
   */
  window.chotgorAutosave = function (formOrSelector) {
    var form =
      typeof formOrSelector === "string"
        ? document.querySelector(formOrSelector)
        : formOrSelector;
    if (form) form.dispatchEvent(new CustomEvent("autosave:now"));
  };
})();
